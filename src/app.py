import requests
import io
import numpy as np
from PIL import Image
import pyarmnn

# Model downloaded locally on the container during docker build
labels_filename = '/onnx_model/synset.txt'
model_filename = '/onnx_model/resnet50-v1-7.onnx'


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def create_network(model_file: str):
    options = pyarmnn.CreationOptions()
    runtime = pyarmnn.IRuntime(options)

    parser = pyarmnn.IOnnxParser()
    network = parser.CreateNetworkFromBinaryFile(model_file)

    opt_network, _ = pyarmnn.Optimize(
        network,
        [pyarmnn.BackendId('CpuAcc')],
        runtime.GetDeviceSpec(),
        pyarmnn.OptimizerOptions()
    )
    net_id, _ = runtime.LoadNetwork(opt_network)

    return net_id, parser, runtime


def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [label.rstrip() for label in f]
        return labels


def download_file(img_url):
    response = requests.get(img_url)
    img_raw = response.content
    img = Image.open(io.BytesIO(img_raw))

    return img


def preprocess(img):
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    img = img.crop((16, 16, 240, 240))
    img = img.convert('RGB')
    img = np.array(img)

    mean = np.array([0.485, 0.456, 0.406])
    stddev = np.array([0.229, 0.224, 0.225])
    img = ((img / 255.0) - mean) / stddev

    img = np.transpose(img, [2, 0, 1])
    img = img.flatten().astype(np.float32)

    return img


def postprocess(scores):
    scores = np.squeeze(scores)
    return softmax(scores)


def inference(network, input_data):
    net_id, parser, runtime = network

    input_binding_info = parser.GetNetworkInputBindingInfo("data")

    output_binding_info = parser.GetNetworkOutputBindingInfo("resnetv17_dense0_fwd")

    input_tensors = pyarmnn.make_input_tensors(
        [input_binding_info],
        [input_data]
    )

    output_tensors = pyarmnn.make_output_tensors([output_binding_info])

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_data = pyarmnn.workload_tensors_to_ndarray(output_tensors)[0][0]

    return output_data


def top_class(N, probs, labels):
    results = np.argsort(probs)[::-1]

    classes = []
    for i in range(N):
        classes.append({
            "class": labels[results[i]],
            "prob": float(probs[results[i]])
        })

    return classes


# Lambda function handler expects an image (URL) that will be used for ML inference
def handler(event, context):
    image = download_file(event["image_url"])
    input_data = preprocess(image)

    network = create_network(model_filename)
    scores = inference(network, input_data)
    probs = postprocess(scores)

    labels = load_labels(labels_filename)
    response = top_class(5, probs, labels)

    return response
