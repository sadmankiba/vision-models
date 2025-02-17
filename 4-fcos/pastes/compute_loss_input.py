
targets = [
    {'boxes': tensor([[141.7920,  30.9895, 336.0000, 174.4842],
        [  0.0000,  88.9263,  50.4000, 212.8842],
        [135.7440, 109.8105, 165.3120, 212.8842],
        [112.2240,  94.3158, 145.8240, 210.1895],
        [ 84.6720,  97.6842, 120.2880, 210.1895],
        [ 59.8080, 101.7263,  90.0480, 211.5368],
        [ 57.1200, 101.7263,  77.2800, 215.5789],
        [ 35.6160,  94.3158,  62.4960, 220.9684]], device='cuda:0'), 
    'labels': tensor([ 3,  6, 14, 14, 14, 14, 14, 14], device='cuda:0'), 
    'image_id': tensor([2005], device='cuda:0'), 
    'area': tensor([27867.8301,  6247.4775,  3047.6819,  3893.3567,  4006.9883,  3320.6702, 2295.2690,  3404.4224], device='cuda:0'), 
    'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
    }, 
    {'boxes': tensor([[  0.0000,  36.2985, 382.5540, 298.9851],
        [374.9220, 122.2687, 418.8060, 228.2985],
        [437.8860, 144.2388, 474.1380, 237.8507],
        [427.3920, 125.1343, 467.4600, 169.0746]], device='cuda:0'), 
    'labels': tensor([18, 14, 14, 14], device='cuda:0'), 
    'image_id': tensor([3197], device='cuda:0'), 
    'area': tensor([100491.7969,   4653.0142,   3393.6213,   1760.5994], device='cuda:0'), 
    'iscrowd': tensor([0, 0, 0, 0], device='cuda:0')
    }, 
    {'boxes': tensor([[  9.6720,   9.6366, 280.4880, 244.6221]], device='cuda:0'), 
    'labels': tensor([14], device='cuda:0'), 
    'image_id': tensor([3703], device='cuda:0'), 
    'area': tensor([63637.8242], device='cuda:0'), 'iscrowd': tensor([0], device='cuda:0')
    }, 
    {'boxes': tensor([[213.5040,  53.7600, 316.4160, 277.2480],
        [ 38.4000,  71.4240, 152.0640, 277.2480],
        [277.2480,  41.4720, 292.6080,  90.6240],
        [304.1280,  49.9200, 318.7200,  82.1760],
        [311.0400,  59.1360, 324.8640,  85.2480],
        [321.0240,  62.9760, 339.4560,  95.2320],
        [ 92.1600, 138.2400, 250.3680, 283.3920],
        [293.3760, 113.6640, 360.1920, 160.5120],
        [321.0240,  96.7680, 384.0000, 127.4880],
        [277.2480,  81.4080, 326.4000, 112.1280],
        [372.4800,  66.8160, 384.0000, 105.9840],
        [259.5840, 116.7360, 374.0160, 285.6960],
        [  0.0000, 122.1120, 118.2720, 287.2320],
        [242.6880, 162.0480, 384.0000, 288.0000],
        [  0.0000, 163.5840,  39.1680, 288.0000]], device='cuda:0'), 
    'labels': tensor([14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 14,  8,  8,  8,  8],
       device='cuda:0'), 
    'image_id': tensor([2434], device='cuda:0'), 
    'area': tensor([22999.6016, 23394.7793,   754.9740,   470.6799,   360.9724,   594.5418,
        22964.2051,  3130.1968,  1934.6223,  1509.9489,   451.2149, 19334.4297,
        19529.0723, 17798.5273,  4873.1260], device='cuda:0'), 
    'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')}]

points = [tensor([[[  4.,   4.],
         [  4.,  12.],
         [  4.,  20.],
            ...,
         [316., 460.],
         [316., 468.],
         [316., 476.]]], device='cuda:0'), tensor([[[  8.,   8.],
         [  8.,  24.],
         [  8.,  40.],
         ...,
         [  8., 440.],
         [  8., 456.],
         [  8., 472.]],
         ...,
         [312., 440.],
         [312., 456.],
         [312., 472.]]], device='cuda:0'), tensor([[[ 16.,  16.],
         [ 16.,  48.],
         [ 16.,  80.],
         [ 16., 112.],
         ...,
         [304., 368.],
         [304., 400.],
         [304., 432.],
         [304., 464.]]], device='cuda:0'), tensor([[[ 32.,  32.],
         [ 32.,  96.],
         [ 32., 160.],
         [ 32., 224.],
         [ 32., 288.],
         
         [288., 416.],
         [288., 480.]]], device='cuda:0'), tensor([[[ 64.,  64.],
         [ 64., 192.],
         [ 64., 320.],
         [ 64., 448.]],

        [[192.,  64.],
         [192., 192.],
         [192., 320.],
         [192., 448.]],

        [[320.,  64.],
         [320., 192.],
         [320., 320.],
         [320., 448.]]], device='cuda:0')]

len(points) = 5       
points[0].shape = torch.Size([40, 60, 2])

points[1].shape = torch.Size([20, 30, 2])
points[2].shape = torch.Size([10, 15, 2])
points[3].shape = torch.Size([5, 8, 2])
points[4].shape = torch.Size([3, 4, 2])

strides = tensor([  8.,  16.,  32.,  64., 128.], device='cuda:0')

reg_range = tensor([[   0.,   64.],
        [  64.,  128.],
        [ 128.,  256.],
        [ 256.,  512.],
        [ 512., 1000.]], device='cuda:0')

len(cls_logits) = 5

cls_logits[0].shape = torch.Size([4, 256, 40, 60])

cls_logits[1].shape = torch.Size([4, 256, 20, 30])

cls_logits[2].shape = torch.Size([4, 256, 10, 15])

cls_logits[3].shape = torch.Size([4, 256, 5, 8])

cls_logits[4].shape = torch.Size([4, 256, 3, 4])

len(reg_outputs) = 5

reg_outputs[0].shape = torch.Size([4, 256, 40, 60])
reg_outputs[1].shape = torch.Size([4, 256, 20, 30])

len(ctr_logits) = 5

ctr_logits[0].shape = torch.Size([4, 256, 40, 60])