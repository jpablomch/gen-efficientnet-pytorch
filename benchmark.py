import argparse
from time import time
import torch
import geffnet

parser = argparse.ArgumentParser(description='benchmark')
parser.add_argument('--mkldnn', action='store_true', default=False,
                    help='use mkldnn blocked memory format')
parser.add_argument('--profile', action='store_true', default=False,
                    help='do profiling')
args = parser.parse_args()

### Issues:
### 1. activation from these models are not covered by _mkldnn memory format
###    efficient_net: SwishJit
###    mobilenet_v2: ReLU6
### 2. adaptive_avg_pool2d from efficient_net is not covered by _mkldnn memory format
###    this is not a perf bottleneck, but breaks up the consistency of _mkldnn format
###
### Solutions:
### 1. provide support of _mkldnn for these OPs, we can
###    a) ask MKLDNN team support them (longer process: mkl-dnn -> ideep -> pytorch upstream)
###    b) hack in aten (shorter process: pytorch upstream; but fb may concern this manner)
### 2. optimize native aten OPs with TensorIterator
###    take channels_last memory format as "turbo mode"
###    upstream path is clear, conv2d perf depends on mkl-dnn NHWC development
###


niters = 1000
nwarmups = int(niters / 100)

tests = ['efficientnet_b0', 'mobilenetv2_100']

def run_single_test(model_name, input_size):
    input = torch.randn(input_size)

    # create model from geffnet
    model = geffnet.create_model(model_name, pretrained=True)
    model.eval()
    # print(model)

    with torch.no_grad():
        for i in range(nwarmups):
            output = model(input)

        t1 = time()
        with torch.autograd.profiler.profile(enabled=args.profile) as prof:
            for i in range(niters):
                output = model(input)

        if args.profile:
            print("\n\n{} profiling result:".format(model_name))
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

        t2 = time()

    ttime = (t2 - t1) / niters * 1000
    print('Model: {}; Input size: {}; time: {:.3f} ms'.format(model_name, input_size, ttime))

for t in tests:
    run_single_test(t, (1, 3, 224, 224))
