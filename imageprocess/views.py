from __future__ import division
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from imageprocess.models import PicTest, UserInfo
import os
import pickle
import cv2
import numpy as np
import sys
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers


def uploadtest(request):
    if request.FILES:
        str_info = ''
        pic = request.FILES.get('pic')
        sava_path = '%s/test/%s' % (settings.MEDIA_ROOT, pic.name)
        with open(sava_path, 'wb') as f:
            for content in pic.chunks():
                f.write(content)
        print('-->图片上传成功...')
        str_info += '-->图片上传成功...\n'
        # 以下是测试过程：
        sys.setrecursionlimit(40000)

        config_output_filename = os.path.join(settings.CONFIG_BISHE, 'bishe/config.pickle')
        #  print(config_output_filename)
        print('-->正在检测...')
        str_info += '-->正在检测...\n'
        with open(config_output_filename, 'rb') as f_in:
            C = pickle.load(f_in)
            print('-->找到配置文件...')

        C.model_path = os.path.join(settings.CONFIG_BISHE, C.model_path)
        print('-->找到模型信息...')
        str_info += '-->找到模型信息...\n'
        print('-->模型路径地址：' + C.model_path)
        str_info += '-->模型路径地址：' + C.model_path + '\n'
        if C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif C.network == 'vgg':
            import keras_frcnn.vgg as nn

        # turn off any data augmentation at test time
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False
        C.num_rois = 10

        def format_img_size(img, C):
            """ formats the image size based on config """
            img_min_side = float(C.im_size)
            (height, width, _) = img.shape

            if width <= height:
                ratio = img_min_side / width
                new_height = int(ratio * height)
                new_width = int(img_min_side)
            else:
                ratio = img_min_side / height
                new_width = int(ratio * width)
                new_height = int(img_min_side)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return img, ratio

        def format_img_channels(img, C):
            """ formats the image channels based on config """
            img = img[:, :, (2, 1, 0)]
            img = img.astype(np.float32)
            img[:, :, 0] -= C.img_channel_mean[0]
            img[:, :, 1] -= C.img_channel_mean[1]
            img[:, :, 2] -= C.img_channel_mean[2]
            img /= C.img_scaling_factor
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img

        def format_img(img, C):
            """ formats an image for model prediction based on config """
            img, ratio = format_img_size(img, C)
            img = format_img_channels(img, C)
            return img, ratio

        # Method to transform the coordinates of the bounding box to its original size
        def get_real_coordinates(ratio, x1, y1, x2, y2):

            real_x1 = int(round(x1 // ratio))
            real_y1 = int(round(y1 // ratio))
            real_x2 = int(round(x2 // ratio))
            real_y2 = int(round(y2 // ratio))

            return (real_x1, real_y1, real_x2, real_y2)

        class_mapping = C.class_mapping

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}

        print('-->乳腺癌病症种类：' + str(class_mapping))
        str_info += '-->乳腺癌病症种类：' + str(class_mapping) + '\n'

        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

        if C.network == 'resnet50':
            num_features = 1024
        elif C.network == 'vgg':
            num_features = 512

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping),
                                   trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('-->由 {} 加载权重信息...'.format(C.model_path))
        str_info += '-->由 {} 加载权重信息...'.format(C.model_path) + '\n'
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        all_imgs = []

        classes = {}

        bbox_threshold = 0.8

        st = time.time()

        filepath = sava_path

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                        overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (
                    int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))
                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        timeused = (time.time() - st)
        print('-->检测完成，用时： {}...'.format(timeused))
        str_info += '-->检测完成，用时： {}'.format(timeused) + '\n'
        aaa = str(all_dets)
        print('-->检测结果：' + aaa)
        str_info += '-->检测结果：' + aaa + '\n'
        result_path = '%s/result/%s' % (settings.MEDIA_ROOT, pic.name)
        cv2.imwrite(result_path, img)
        print(str_info)
        # 将图片路径回传，为上传数据库做准备
        test_pic = '/static/media/test/%s' % (pic.name)
        result_pic = '/static/media/result/%s' % (pic.name)
        user_id = UserInfo.objects.get(username=request.session.get('username')).id
        return JsonResponse(
            {'res': 1, 'result_pic': result_pic, 'test_pic': test_pic, 'user_id': user_id, 'str_info': str_info})
    return JsonResponse({'res': 0})


def savepic(request):
    name = request.POST.get('name')
    age = request.POST.get('age')
    gender = request.POST.get('gender')
    test_pic = request.POST.get('test_pic')
    result_pic = request.POST.get('result_pic')
    user_id = request.POST.get('user_id')
    result = request.POST.get('info')

    new_pictest = PicTest()
    new_pictest.name = name
    new_pictest.age = age
    new_pictest.gender = gender
    new_pictest.test_pic = test_pic
    new_pictest.result_pic = result_pic
    new_pictest.user_id = user_id
    new_pictest.result = result
    new_pictest.save()
    return JsonResponse({'res': 1})


def showlist(request):
    if request.session.has_key('islogin'):
        username = request.session.get('username')
        user = UserInfo.objects.get(username=username)
        pics = PicTest.objects.filter(user_id=user.id)

        return render(request, 'imageprocess/detail.html', {'username': username, 'pics': pics})

    return redirect('/index/')
