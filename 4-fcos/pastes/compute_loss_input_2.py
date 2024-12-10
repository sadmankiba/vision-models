targets = [
    {'boxes': tensor([[ 13.6640,  65.8824, 379.1760, 241.2834],
        [  0.0000,  88.1283, 122.9760, 181.3904]], device='cuda:0'), 
     'labels': tensor([6, 6], device='cuda:0'), 
     'image_id': tensor([2198], device='cuda:0'), 
     'area': tensor([64111.1914, 11468.9902], device='cuda:0'), 
     'iscrowd': tensor([0, 0], device='cuda:0')
     }, 
    {'boxes': tensor([[ 37.8125,  71.3462, 275.0000, 262.7462],
        [124.4375,  46.6495, 221.3750, 178.3656]], device='cuda:0'), 'labels': tensor([12, 14], device='cuda:0'), 'image_id': tensor([1903], device='cuda:0'), 'area': tensor([45397.6914, 12768.2314], device='cuda:0'), 'iscrowd': tensor([0, 0], device='cuda:0')}, {'boxes': tensor([[  4.7600, 163.8095, 129.4720, 197.1429],
        [  0.0000, 176.1905,  48.5520, 225.7143],
        [102.8160, 193.3333, 248.4720, 235.2381]], device='cuda:0'), 'labels': tensor([5, 6, 6], device='cuda:0'), 'image_id': tensor([4052], device='cuda:0'), 'area': tensor([4157.0659, 2404.4797, 6103.6816], device='cuda:0'), 'iscrowd': tensor([0, 0, 0], device='cuda:0')}, {'boxes': tensor([[  0.0000, 110.5920, 185.0880, 190.4640],
        [332.5440, 169.7280, 344.0640, 194.3040],
        [244.9920, 159.7440, 252.6720, 188.1600],
        [221.9520, 155.9040, 228.8640, 179.7120],
        [214.2720, 155.1360, 219.6480, 177.4080],
        [207.3600, 165.8880, 215.8080, 179.7120]], device='cuda:0'), 'labels': tensor([18, 14, 14, 14, 14, 14], device='cuda:0'), 'image_id': tensor([3825], device='cuda:0'), 'area': tensor([14783.3486,   283.1153,   218.2347,   164.5609,   119.7341,   116.7852],
       device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0, 0], device='cuda:0')}]

len(points)
5

points[0].shape
torch.Size([40, 60, 2])

points[1].shape
torch.Size([20, 30, 2])

...

len(cls_logits)
5

cls_logits[0].shape
torch.Size([4, 20, 60, 52])

cls_logits[1].shape
torch.Size([4, 20, 30, 26])

cls_logits[2].shape
torch.Size([4, 20, 15, 13])

cls_logits[3].shape
torch.Size([4, 20, 8, 7])

cls_logits[4].shape
torch.Size([4, 20, 4, 4])

len(reg_outputs) = 5

reg_outputs[0].shape
torch.Size([4, 4, 40, 60])

 p reg_outputs[1].shape
torch.Size([4, 4, 20, 30])

...

len(ctr_logits) = 5

ctr_logits[0].shape = torch.Size([4, 1, 40, 60])

...