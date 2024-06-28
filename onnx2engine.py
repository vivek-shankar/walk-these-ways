import tensorrt as trt


def build_engine(onnx_file_path, engine_file_path):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    builder.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_cuda_engine(network)

    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    return engine


label = "gait-conditioned-agility/pretain-a1/train/081719.645276"
dirs = glob.glob(f"../../runs/{label}/*")
logdir = sorted(dirs)[0]
print(logdir)

body_onnx_file_path = logdir + '/onnx_models/body_model.onnx'
body_engine_file_path = logdir + '/engines/body_model.trt'
adaption_onnx_file_path = logdir + '/onnx_models/adaptation_module_model.onnx'
adaption_engine_file_path = logdir + '/engines/adaptation_module_model.trt'
build_engine(body_onnx_file_path, body_engine_file_path)
build_engine(adaption_onnx_file_path, adaption_engine_file_path)
print('engines have been build')
