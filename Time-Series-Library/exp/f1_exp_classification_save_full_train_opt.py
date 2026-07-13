from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification_Save_FullTrainOpt(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification_Save_FullTrainOpt, self).__init__(args)
        print("\n========== Experiment Arguments ==========")
        for k, v in sorted(vars(args).items()):
            print(f"{k:25}: {v}")
        print("==========================================\n")



    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        # ✅ Freeze everything except the classifier head
        for name, param in model.named_parameters():
            param.requires_grad = True
            print(f"Unfrozen: {name}")

                
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, class_weights=None):
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_probs = []
        all_trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                # ✅ collect probs instead of argmax
                probs = torch.nn.functional.softmax(pred, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_trues.extend(label.flatten().cpu().numpy())

        total_loss = np.average(total_loss)
        all_probs = np.array(all_probs)
        all_trues = np.array(all_trues)

        # ✅ threshold sweep — identical to LSTM/Transformer NAS
        from sklearn.metrics import f1_score
        best_val_f1 = -1.0
        for thr in np.linspace(0.2, 0.8, 61):
            preds = (all_probs >= thr).astype(int)
            f1 = f1_score(all_trues, preds, zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1

        self.model.train()
        return total_loss, best_val_f1

    def train(self, setting):
        print("\n✅ Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(" -", name)

        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        # ================= DEBUG: LABEL CHECK =================
        print("🧪 DEBUG STARTED")
        y_train = train_data.labels_df.values.flatten()
        print("🧪 DEBUG labels unique:", np.unique(y_train))
        print("🧪 DEBUG labels min/max:", y_train.min(), y_train.max())
        print("🧪 DEBUG num classes:", self.args.num_class)
        y_val = vali_data.labels_df.values.flatten()
        print("🧪 DEBUG val")
        print("🧪 DEBUG labels unique:", np.unique(y_val))
        print("🧪 DEBUG labels min/max:", y_val.min(), y_val.max())
        print("🧪 DEBUG num classes:", self.args.num_class)
        # =====================================================
        # ============================================================
        # 🔥 CLASS IMBALANCE HANDLING (ADD THIS)
        # ============================================================
        from sklearn.utils.class_weight import compute_class_weight

        # training labels (numpy array)
        y_train = train_data.labels_df.values.flatten()

        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        print("⚖️ Using class weights:", class_weights.tolist())

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        #criterion = self._select_criterion()
        criterion = self._select_criterion(class_weights)

        

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            #vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            #test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            #print(
            #    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
             #   .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            #early_stopping(-val_accuracy, self.model, path)
            vali_loss, val_f1 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_f1 = self.vali(test_data, test_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali F1: {4:.3f} Test Loss: {5:.3f} Test F1: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_f1, test_loss, test_f1))
            early_stopping(-val_f1, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # ✅ ADD THIS — save VAL probabilities for NAS scoring
        self.model.eval()
        val_probs_all, val_trues_all = [], []
        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                val_probs_all.extend(probs.cpu().numpy())
                #val_trues_all.extend(label.flatten().numpy())
                val_trues_all.extend(label.flatten().cpu().numpy())


        save_path = os.path.join('./test_results', setting)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'val_prob.npy'), np.array(val_probs_all))
        np.save(os.path.join(save_path, 'val_true.npy'), np.array(val_trues_all))
        print(f"✅ Saved VAL probabilities for NAS scoring")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            # ✅ 1. Load the checkpoint into a variable first
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

            # ✅ 2. Remove projection layers that don't match current model
            bad_keys = [k for k in state_dict.keys() if "projection" in k]
            for k in bad_keys:
                print(f"Skipping incompatible layer: {k}")
                del state_dict[k]

            # ✅ 3. Load the remaining weights safely
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded pretrained weights (except projection).")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        #probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        probs = torch.nn.functional.softmax(preds, dim=1)  # ✅ specify dim=1 for clarity
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        if probs.shape[1] == 2:
            probs_np = probs[:, 1].cpu().numpy()  # probability of class 1 (binary classification)
        else:
            probs_np = probs.cpu().numpy()  # all class probabilities (for multi-class AUC)


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        # save metrics
        file_name = 'result_classification.txt'
        with open(os.path.join(folder_path, file_name), 'a') as f:
            f.write(setting + "  \n")
            f.write(f'accuracy:{accuracy}\n\n')

        # ✅ Also save raw predictions and labels
        import numpy as np
        save_path = os.path.join('./test_results', setting)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'pred.npy'), predictions)
        np.save(os.path.join(save_path, 'true.npy'), trues)
        np.save(os.path.join(save_path, 'prob.npy'), probs_np)  # ✅ save probabilities for AUC
        print(f"✅ Saved predictions and true labels to {save_path}")
        return
