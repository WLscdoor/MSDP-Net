import torch
import torch.nn as nn
from thop import profile
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from myutils.save_and_load_model import save_checkpoint, reload_best_checkpoint, print_the_para, freeze_and_unfreeze
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from myutils.confusion_matrix import draw_confusion_matrix, draw_tsne


def trainAndTest_model(model, train_dataset, val_dataset, test_dataset, args, device):

    # **************************************************************
    # para setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.005)
    StepLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(args.log_tensorboard_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*4, shuffle=False)
    epoch_i = 0
    loss_values = []
    best_loss = float('inf')
    total_step = 0
    start_time = 0
    end_time = 0
    # **************************************************************
    # start training
    for epoch in range(epoch_i, args.epoch_num):
        model.train()
        current_loss = 0
        step_itr = 0
        # end_time = time.time()
        for batch_idx, (HRRP_one, HRRP_label, label, filename) in enumerate(train_loader):
            optimizer.zero_grad()
            HRRP_one, HRRP_label, label = HRRP_one.to(device).to(torch.float32), HRRP_label.to(device).to(torch.float32), label.long().to(device)

            outputs, _ = model(HRRP_one)

            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            step_itr += 1            
            total_step +=1
            writer.add_scalar('loss', loss.item(), total_step)
            loss_values.append(loss.item())
        current_loss /= step_itr
        end_time = time.time()
        writer.add_scalar('train_epoch_loss', current_loss, epoch)
        StepLR.step()
        print(f'Epoch [{epoch + 1}/{args.epoch_num}] completed. Current train loss = [{current_loss}], Runtime = [{end_time-start_time}]')
        # **************************************************************
        # evaluate the model with val dataset
        model.eval()
        all_labels = []
        predicted_labels = []
        current_loss = 0
        step_itr = 0
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (HRRP_one, HRRP_label, label, filename) in enumerate(val_loader):
                HRRP_one, HRRP_label, label = HRRP_one.to(device).to(torch.float32), HRRP_label.to(device).to(torch.float32), label.long().to(device)
                outputs, _ = model(HRRP_one)
                loss = criterion(outputs, label)
                current_loss += loss.item()
                step_itr += 1
        # **************************************************************
        # save the best model
        current_loss /= step_itr
        writer.add_scalar('val_epoch_loss', current_loss, epoch)
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, args, epoch)
            print(
                f'New best val loss model saved at Epoch {epoch + 1},  with loss: {best_loss:.4f}')
    # **************************************************************
    # evaluate the best model with test dataset
    # best_epoch = 34
    model = reload_best_checkpoint(args, model ,best_epoch)
    model.eval()
    labels_name = {target: i for i, target in enumerate(["costa", "Boat7", "lng", "patrol","transport3","container1","container2","owlfelino"])}
    all_labels = []
    predicted_labels = []
    current_loss = 0
    step_itr = 0
    x_sne = []
    with torch.no_grad():
        for batch_idx, (HRRP_one, HRRP_label, label, filename) in enumerate(test_loader):
            HRRP_one, HRRP_label, label = HRRP_one.to(device).to(torch.float32), HRRP_label.to(device).to(torch.float32), label.long().to(device)
            outputs, fea  = model(HRRP_one)
            # loss_info = info_loss(proj, org_fea)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            current_loss += loss.item()
            x_sne.extend(fea.cpu().numpy())
            step_itr += 1
    
    # **************************************************************
    # calculate the FLOPs, params, and the runtime
    current_loss /= step_itr 
    flops, params = profile(model, inputs=(HRRP_one,))
    print(f'FLOPs: {flops / 1e9} GFLOPs')
    print(f'Params: {params / 1e6} M')
    start = time.time()
    _ = model(HRRP_one)
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")
    # **************************************************************
    # calculate and save the T-SNE
    draw_tsne(x_sne, all_labels, labels_name)
    # **************************************************************
    # calculate and save the metrics
    accuracy = 100 * np.sum(np.array(predicted_labels) == np.array(all_labels)) / len(all_labels)
    print(f'All epoch completed. Best epoch = {best_epoch}. Current test loss = [{current_loss}]')
    print_the_para(args.log_txt_dir, f'Best epoch: {best_epoch:.2f}%')
    print_the_para(args.log_txt_dir, f'loss on test set: {current_loss:.2f}%')

    print(f'Accuracy on test set: {accuracy:.2f}%')
    print_the_para(args.log_txt_dir, f'Accuracy on test set: {accuracy:.2f}%')

    f1score = f1_score(y_true=all_labels, y_pred=predicted_labels, average='macro')
    print(f'F1 Score: {100*f1score:.2f}%')
    print_the_para(args.log_txt_dir, f'F1 Score: {f1score}')

    precision = precision_score(y_true=all_labels, y_pred=predicted_labels, average='macro')
    recall = recall_score(y_true=all_labels, y_pred=predicted_labels, average='macro')
    print(f'Precision: {100*precision:.2f}%')
    print(f'Recall: {100*recall:.2f}%')
    print_the_para(args.log_txt_dir, f'Precision: {precision}')
    print_the_para(args.log_txt_dir, f'Recall: {recall}')
    # print(classification_report(all_labels, predicted_labels, digits=4))

    label_to_count = {i: 0 for i in range(len(train_dataset.labels))}
    correct_predictions_per_label = {i: 0 for i in range(len(train_dataset.labels))}

    for true_label, pred_label in zip(all_labels, predicted_labels):
        label_to_count[true_label] += 1
        if true_label == pred_label:
            correct_predictions_per_label[true_label] += 1

    class_accuracies = {label: (correct / count) * 100 for label, correct, count in zip(
        correct_predictions_per_label.keys(), correct_predictions_per_label.values(), label_to_count.values()) if
                        count > 0}

    for label, accuracy in class_accuracies.items():
        print(f"Class {label}: Accuracy = {accuracy:.2f}%")
        print_the_para(args.log_txt_dir, f"Class {label}: Accuracy = {accuracy:.2f}%")

    # calculate and save the confusion matrix
    draw_confusion_matrix(label_true=all_labels, label_pred= predicted_labels, label_name=labels_name)
