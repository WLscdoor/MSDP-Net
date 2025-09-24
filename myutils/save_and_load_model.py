import torch
import os
import datetime

def save_checkpoint(model, optimizer, args, epoch_now):
    torch.save({'state_dict': model.state_dict(), 
            'epoch': epoch_now,
            'optimizer': optimizer.state_dict(),
            'args': args},
            os.path.join(args.save_dir, f'ATRnet_total_epoch={args.epoch_num}.pth.tar')
            )

def reload_best_checkpoint(args, model ,best_epoch):
    path = os.path.join(args.save_dir, f'ATRnet_total_epoch={args.epoch_num}.pth.tar')
    assert os.path.isfile(path), 'checkpoint not found.'
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(' Evaluating model from best checkpoint')
    return model


def print_the_para(file_path, para):
    with open(file_path, "a") as file:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{formatted_time}-----\t{para}\n")


def freeze_and_unfreeze(model):
    for param in model.parameters():
        param.requires_grad = False  
    for param in model.fc.parameters():  
        param.requires_grad = True
    print("Training Para:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model

def count_parameters(model):

    result = {}
    total = 0
    for name, module in model.named_children():  # 只遍历直接子模块
        param_sum = sum(p.numel() for p in module.parameters())
        result[name] = param_sum
        total += param_sum
    result['Total'] = total
    for name, count in result.items():
        print(f"{name}: {count/ 1e6} M")
