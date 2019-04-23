import json
import numpy as np
import os
import torch
import types

from torchvision import transforms
from PIL import Image, ImageDraw

from .model import resnet34

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = json.load(fp)
    return names


def label_to_class(self, label: int) -> str:
    """Returns string of class name given index
    """
    return self.classes[str(label)]


def class_to_label(self, _class: str) -> int:
    """Returns string of class name given index
    """
    return int(list(self.classes.keys())[list(self.classes.values()).index(_class)])


def clamp(n, minn, maxn):
    """Make sure n is between minn and maxn

    Args:
        n (number): Number to clamp
        minn (number): minimum number allowed
        maxn (number): maximum number allowed
    """
    return max(min(maxn, n), minn)


def postprocess(self, detections: torch.tensor, input_img: Image, visualize: bool = False):
    """Converts PyTorch tensor into interpretable format

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the Rocket family there might be additional options.
    This model supports either outputting a list of bounding boxes of the format
    (x0, y0, w, h) or outputting a `PIL.Image` with the bounding boxes
    and (class name, class confidence, object confidence) indicated.

    Args:
        detections (Tensor): Output Tensor to postprocess
        input_img (PIL.Image): Original input image which has not been preprocessed yet
        visualize (bool): If True outputs image with annotations else a list of bounding boxes
    """

    img = np.array(input_img)
    img_height, img_width, _ =  img.shape

    predicted_classes = torch.max(detections, 1)[1]

    # best_detection = list({
    #             'topLeft_x': 1,
    #             'topLeft_y': 1,
    #             'width': img_width-1,
    #             'height': img_height-1,
    #             'bbox_confidence': bbox_confidence,
    #             'class_name': class_name,
    #             'class_confidence': class_confidence})

    best_detection = [{
                'class_index': np.array(predicted_classes)[0],
                'class_name': self.label_to_class(label=np.array(predicted_classes)[0])
                }]

    if visualize:
        line_width = 2
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        # Extract information from the detection
        topLeft = (5, 5)
        bottomRight = (img_width-10, img_height-10)
        # Draw the bounding boxes and the information related to it        
        ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
        class_index = best_detection[0]['class_index']        
        class_name = best_detection[0]['class_name']
        ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="Index: {} -- Class: {}".format(class_index, class_name))
        del ctx
        return img_out

    return best_detection


def train_forward(self, x: torch.Tensor, targets: torch.Tensor):
    """Performs forward pass and returns loss of the model

    The loss can be directly fed into an optimizer.
    """
    self.training = True
    loss = self.forward((x, targets))
    self.training = False
    return loss


def preprocess(self, img: Image, labels: list = None) -> torch.Tensor:
    """Converts PIL Image or Array into PyTorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """

    # todo: support batch size bigger than 1 for training and inference
    # todo: replace this hacky solution and work directly with tensors
    if type(img) == Image.Image:
        # PIL.Image
        pass
    elif type(img) == torch.Tensor:
        # list of tensors
        img = img[0].cpu()
        img = transforms.ToPILImage()(img)
    elif "PIL" in str(type(img)): # type if file just has been opened
        img = img.convert("RGB")
    else:
        raise TypeError("wrong input type: got {} but expected list of PIL.Image, "
                        "single PIL.Image or torch.Tensor".format(type(img)))

    new_w, new_h, pad_w, pad_h, _ = get_new_size_and_padding(img)

    input_transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    padding = torch.nn.ConstantPad2d((pad_h//2, pad_h//2, pad_w//2, pad_w//2), 0.0)

    out_tensor = input_transform(img).unsqueeze(0)
    out_tensor = padding(out_tensor)

    if labels is None:
        return out_tensor

    # if type(img) == list:
    #     out_tensor = None
    #     for elem in img:
    #         out = input_transform(elem).unsqueeze(0)
    #         if out_tensor is not None:
    #             torch.cat((out_tensor, out), 0)
    #         else:
    #             out_tensor = out
    # else:
    #     out_tensor = input_transform(img).unsqueeze(0)

    max_objects = 50
    filled_labels = np.zeros((max_objects, 5))  # max objects in an image for training=50, 5=(x1,y1,x2,y2,category_id)
    if labels is not None:
        for idx, label in enumerate(labels):

            # add padding
            label[0] += pad_w//2
            label[1] += pad_h//2
            label[2] += pad_w // 2
            label[3] += pad_h // 2

            padded_w = new_w + pad_w
            padded_h = new_h + pad_h

            # resize coordinates to match Yolov3 input size
            scale_x = new_w / padded_w
            scale_y = new_h / padded_h

            label[0] *= scale_x
            label[1] *= scale_y
            label[2] *= scale_x
            label[3] *= scale_y

            x1 = label[0]
            y1 = label[1]
            x2 = label[2]
            y2 = label[3]

            # x1 = label[0] / new_w
            # y1 = label[1] / new_h
            #
            # cw = (label[2]) / new_w
            # ch = (label[3]) / new_h
            #
            # cx = (x1 + (x1 + cw)) / 2.0
            # cy = (y1 + (y1 + ch)) / 2.0

            filled_labels[idx] = np.asarray([x1, y1, x2, y2, label[4]])
            if idx >= max_objects:
                break
    filled_labels = torch.from_numpy(filled_labels)

    return out_tensor, filled_labels.unsqueeze(0)


def build():
    classes = load_classes(os.path.join(os.path.realpath(os.path.dirname(__file__)), "imagenet-1000.json"))
    
    model = resnet34(num_classes=len(classes), pretrained=False)

    model.load_state_dict(torch.load(os.path.join(os.path.realpath(os.path.dirname(__file__)), "weights.pth"),
                                     map_location=torch.device('cpu')))

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    model.label_to_class = types.MethodType(label_to_class, model)
    model.train_forward = types.MethodType(train_forward, model)
    setattr(model, 'classes', classes)
    setattr(model, 'num_classes', len(classes))

    return model