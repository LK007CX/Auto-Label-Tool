from PyQt5.QtCore import QThread, pyqtSignal,  QSize
from PyQt5.QtWidgets import QLabel, QMainWindow, QProgressBar, QFileDialog, QApplication
from sys import argv, exit
from YM import Ui_MainWindow
from os import getcwd, path, listdir
from cv2 import imread, imwrite, warpAffine, flip, getRotationMatrix2D, INTER_LANCZOS4, boundingRect
from math import ceil
from numpy import float32, deg2rad, sin, cos, dot, array, vstack, int32
from copy import deepcopy
from skimage import exposure
from skimage.util import random_noise
from random import randint, random, uniform
from datetime import datetime
import xml.etree.ElementTree as ET
from time import sleep
import threading

StyleSheet = '''
#statusLabel {
    font-family: "Microsoft YaHei";
}
'''


class Winform(QMainWindow):

    def __init__(self, parent=None):
        super(Winform, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("YOLO图像增广工具")
        self.progressBar = QProgressBar()
        self.progressBar.setMaximumWidth(250)
        self.progressBar.setValue(0)
        self.ui.statusbar.addWidget(self.progressBar)

        self.statusLabel = QLabel(objectName="statusLabel")
        self.statusLabel.setMinimumWidth(150)
        self.ui.statusbar.addPermanentWidget(self.statusLabel)

        self.tipLabel = QLabel()
        self.tipLabel.setText("Tips: 请不要在进度条滚动时点击\"开始增广\"按钮；文件路径不要包含空格")
        self.tipLabel.setMinimumWidth(400)
        self.ui.statusbar.addWidget(self.tipLabel)

        self.initActions()

        #self.resize(QSize(700, 300))

    def initActions(self):
        self.workThread = WorkThread()
        self.workThread.processBar_Signal.connect(self.changeProcessBar)
        self.workThread.enable_startPushButton.connect(self.enable_startPushButton)
        self.workThread.emit_str.connect(self.show_info)
        self.ui.cropHorizontalSlider.valueChanged.connect(self.changeCropText)
        self.ui.translationHorizontalSlider.valueChanged.connect(self.changeTranslationText)
        self.ui.lightHorizontalSlider.valueChanged.connect(self.changeLightText)
        self.ui.noiseHorizontalSlider.valueChanged.connect(self.changeNoiseText)
        self.ui.rotateHorizontalSlider.valueChanged.connect(self.changeRotateText)
        self.ui.lightDOWNDoubleSpinBox.valueChanged.connect(self.changeLightDown)
        self.ui.lightUPDoubleSpinBox.valueChanged.connect(self.changeLightUp)
        self.ui.rotateDOWNdoubleSpinBox.valueChanged.connect(self.changeRotateDown)
        self.ui.rotateUPDoubleSpinBox.valueChanged.connect(self.changeRotateUp)

        self.ui.startPushButton.clicked.connect(self.workThread.thread_work)

        self.ui.oldToolButton.clicked.connect(self.openOldFolder)
        self.ui.newToolButton.clicked.connect(self.openNewFolder)
        self.ui.exitPushButton.clicked.connect(self.exit)
        self.ui.resetPushButton.setVisible(False)
        self.workThread.light_down = self.ui.lightDOWNDoubleSpinBox.value()
        self.workThread.light_up = self.ui.lightUPDoubleSpinBox.value()

        self.workThread.rotate_down = self.ui.rotateDOWNdoubleSpinBox.value()
        self.workThread.rotate_up = self.ui.rotateUPDoubleSpinBox.value()

        print("light down: " + str(self.workThread.light_down))
        print("light up: " + str(self.workThread.light_up))

        print("rotate down: " + str(self.workThread.rotate_down))
        print("rotate up: " + str(self.workThread.rotate_up))

    def show_info(self, val):
        self.statusLabel.setText(val)

    def enable_startPushButton(self):
        pass

    def changeProcessBar(self, value):
        self.progressBar.setValue(value)

    def openOldFolder(self):
        self.workThread.old_folder = QFileDialog.getExistingDirectory(self, "选择原始标注地址", "./")
        self.ui.originalFolderLabel.setText(str(self.workThread.old_folder))

    def openNewFolder(self):
        self.workThread.new_folder = QFileDialog.getExistingDirectory(self, "选择原始标注地址", "./")
        self.ui.changedFolderLabel.setText(str(self.workThread.new_folder))

    def changeCropText(self, value):
        self.ui.cropLabel.setText(str(value))
        self.workThread.crop_num = value
        print("crop num: " + str(self.workThread.crop_num))

    def changeTranslationText(self, value):
        self.ui.translationLabel.setText(str(value))
        self.workThread.translation_num = value
        print("tran num: " + str(self.workThread.translation_num))

    def changeLightText(self, value):
        self.ui.lightLabel.setText(str(value))
        self.workThread.light_num = value
        print("light num: " + str(self.workThread.light_num))

    def changeNoiseText(self, value):
        self.ui.noiseLabel.setText(str(value))
        self.workThread.noise_num = value
        print("noise num: " + str(self.workThread.noise_num))

    def changeRotateText(self, value):
        self.ui.rotateLabel.setText(str(value))
        self.workThread.rotate_num = value
        print("rotate num: " + str(self.workThread.rotate_num))

    def changeLightDown(self, value):
        self.workThread.light_down = round(value, 2)
        self.statusLabel.setText(str(round(value, 2)))
        print("light down: " + str(self.workThread.light_down))

    def changeLightUp(self, value):
        self.workThread.light_up = round(value, 2)
        self.statusLabel.setText(str(round(value, 2)))
        print("light up: " + str(self.workThread.light_up))

    def changeRotateDown(self, value):
        self.workThread.rotate_down = int(value)
        self.statusLabel.setText(str(value))
        print("rotate down: " + str(self.workThread.rotate_down))

    def changeRotateUp(self, value):
        self.workThread.rotate_up = int(value)
        self.statusLabel.setText(str(value))
        print("rotate up: " + str(self.workThread.rotate_up))

 
    def exit(self):
        qApp = QApplication.instance()
        qApp.quit()


class WorkThread(QThread):
    processBar_Signal = pyqtSignal(float)
    enable_startPushButton = pyqtSignal()
    emit_str = pyqtSignal(str)

    def __init__(self, parent=None):
        super(WorkThread, self).__init__(parent)
        # values
        self.crop_num = 0
        self.translation_num = 0
        self.light_num = 0
        self.noise_num = 0
        self.rotate_num = 0

        self.light_down = 0

        self.light_up = 0

        self.rotate_down = 0
        self.rotate_up = 0

        self.to_do_num = 0
        self.total_num = 0

        self.old_folder = ""
        self.new_folder = ""



    def getBoxes(self, image_name):
        """
        根据XML标注文件得到标注列表[x_min, y_min, x_max, y_max, cat_name]的列表
        :param image_name:
        :return:
        """
        tree = ET.parse(image_name + '.xml')
        root = tree.getroot()
        boxes = []
        for object in root.findall('object'):
            temp_list = []
            name = object.find('name').text
            for coordinate in object.find('bndbox'):
                temp_list.append(int(coordinate.text))
            temp_list.append(name)
            boxes.append(temp_list)
        # print(boxes)
        return boxes

    def saveXML(self, save_image_name, save_xml_name, boxes, shape1, shape0):
        folder = ET.Element('folder')
        folder.text = 'image'
        filename = ET.Element('filename')
        filename.text = save_image_name.split('/')[1]
        path = ET.Element('path')
        curr_path = getcwd()
        path.text = curr_path + '/image/' + save_image_name
        source = ET.Element('source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        size = ET.Element('size')
        width = ET.SubElement(size, 'width')
        width.text = str(shape1)
        height = ET.SubElement(size, 'height')
        height.text = str(shape0)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        segmented = ET.Element('segmented')
        segmented.text = '0'
        root = ET.Element('annotation')
        root.extend((folder, filename, path))
        root.extend((source, size, segmented))

        for box in boxes:
            object = ET.Element('object')
            name = ET.SubElement(object, 'name')
            name.text = box[4]
            pose = ET.SubElement(object, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(object, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(box[0])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(box[1])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(box[2])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(box[3])
            root.extend((object,))

        tree = ET.ElementTree(root)
        tree.write(save_xml_name)
        tree = ET.parse(save_xml_name)  # 解析movies.xml这个文件
        root = tree.getroot()  # 得到根元素，Element类
        self.pretty_xml(root, '\t', '\n')  # 执行美化方法
        tree.write(save_xml_name)
        print("Save new XML to:\t" + save_xml_name)
        # self.emit_str.emit("Save new XML to:\t" + save_xml_name)

    def pretty_xml(self, element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素

            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

    def changeImages(self, new_folder, function_name, image_name, n):
        image = imread(image_name + '.jpg')
        boxes = self.getBoxes(image_name)

        if function_name == "flip":
            print("#" + str(self.total_num + 1) + " image enhancement:")
            self.emit_str.emit("#" + str(self.total_num + 1) + " image enhancement")
            print("Old image name:\t" + image_name + ".jpg")
            # self.emit_str.emit("Old image name:\t" + image_name + ".jpg")
            print("Old XML name:\t" + image_name + ".xml")
            # self.emit_str.emit("Old XML name:\t" + image_name + ".xml")
            change_img, change_boxes = self.__flipImage(deepcopy(image), deepcopy(boxes), -1)
            print("Old boxes:\t", boxes)
            # self.emit_str.emit("Old boxes:\t" + str(boxes))
            print("New boxes:\t", change_boxes)
            # self.emit_str.emit("New boxes:\t" + str(change_boxes))
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            current_num = str(randint(0, 9999))
            save_image_name = new_folder + '/' + current_time + '_'  + "-1" + current_num + '.jpg'
            save_xml_name = new_folder + '/' + current_time + '_' + "-1" + current_num + '.xml'
            imwrite(save_image_name, change_img)
            print("Save new image to:\t" + save_image_name)
            # self.emit_str.emit("Save new image to:\t" + save_image_name)
            self.saveXML(save_image_name, save_xml_name, change_boxes, change_img.shape[1], change_img.shape[0])
            print("\n")
            self.total_num += 1
            temp_val = int(self.total_num / self.to_do_num * 100)
            self.processBar_Signal.emit(temp_val)

            print("#" + str(self.total_num + 1) + " image enhancement:")
            self.emit_str.emit("#" + str(self.total_num + 1) + " image enhancement")
            print("Old image name:\t" + image_name + ".jpg")
            # self.emit_str.emit("Old image name:\t" + image_name + ".jpg")
            print("Old XML name:\t" + image_name + ".xml")
            # self.emit_str.emit("Old XML name:\t" + image_name + ".xml")
            change_img, change_boxes = self.__flipImage(deepcopy(image), deepcopy(boxes), 0)
            print("Old boxes:\t", boxes)
            # self.emit_str.emit("Old boxes:\t" + str(boxes))

            print("New boxes:\t", change_boxes)
            # self.emit_str.emit("New boxes:\t" + str(change_boxes))

            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            current_num = str(randint(0, 9999))
            save_image_name = new_folder + '/' + current_time + '_' + "0" + current_num + '.jpg'
            save_xml_name = new_folder + '/' + current_time + '_' + "0" + current_num + '.xml'
            imwrite(save_image_name, change_img)
            print("Save new image to:\t" + save_image_name)
            # self.emit_str.emit("Save new image to:\t" + save_image_name)
            self.saveXML(save_image_name, save_xml_name, change_boxes, change_img.shape[1], change_img.shape[0])
            print("\n")
            self.total_num += 1
            temp_val = int(self.total_num / self.to_do_num * 100)
            self.processBar_Signal.emit(temp_val)
            return

        if function_name == "crop":
            function = self.__cropImage
        elif function_name == "tran":
            function = self.__translationImage
        elif function_name == "light":
            function = self.__changeLightofImage
        elif function_name == "noise":
            function = self.__addNoiseToImage
        elif function_name == "rotate":
            function = self.__rotateImage

        for i in range(1, n + 1):
            print("#" + str(self.total_num + 1) + " image enhancement:")
            self.emit_str.emit("#" + str(self.total_num + 1) + " image enhancement")
            print("Old image name:\t" + image_name + ".jpg")
            # self.emit_str.emit("Old image name:\t" + image_name + ".jpg")
            print("Old XML name:\t" + image_name + ".xml")
            # self.emit_str.emit("Old XML name:\t" + image_name + ".xml")
            change_img, change_boxes = function(deepcopy(image), deepcopy(boxes))
            print("Old boxes:\t", boxes)
            # self.emit_str.emit("Old boxes:\t" + str(boxes))
            print("New boxes:\t", change_boxes)
            # self.emit_str.emit("New boxes:\t" + str(change_boxes))
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            current_num = str(randint(0, 9999))
            save_image_name = new_folder + '/' + current_time + '_' + str(i) + current_num + '.jpg'
            save_xml_name = new_folder + '/' + current_time + '_' + str(i) + current_num + '.xml'
            imwrite(save_image_name, change_img)
            print("Save new image to:\t" + save_image_name)
            # self.emit_str.emit("Save new image to:\t" + save_image_name)
            self.saveXML(save_image_name, save_xml_name, change_boxes, change_img.shape[1], change_img.shape[0])
            print("\n")
            self.total_num += 1
            temp_val = int(self.total_num / self.to_do_num * 100)
            self.processBar_Signal.emit(temp_val)

    def work(self):
        try:
            multiple = self.crop_num + self.translation_num + self.light_num + self.noise_num + self.rotate_num + 2
            image_num = len(listdir(self.old_folder)) / 2
            self.to_do_num = multiple * image_num
            print("self.to_do_num: " + str(self.to_do_num))

            if not path.exists(self.old_folder):
                return
                raise ValueError("Invalid old image folder!")

            if not path.exists(self.old_folder):
                return
                raise ValueError("Invalid old image folder!")
            for filename in listdir(self.old_folder):
                if path.splitext(filename)[1] == '.jpg':  # 目录下包含.jpg的文件
                    old_name = self.old_folder + '/' + path.splitext(filename)[0]
                    self.changeImages(self.new_folder, "crop", old_name, self.crop_num)
                    self.changeImages(self.new_folder, "tran", old_name, self.translation_num)
                    self.changeImages(self.new_folder, "light", old_name, self.light_num)
                    self.changeImages(self.new_folder, "noise", old_name, self.noise_num)
                    self.changeImages(self.new_folder, "rotate", old_name, self.rotate_num)

            print("Save " + str(self.total_num) + " images to " + self.new_folder + ".")
            self.emit_str.emit("Total: " + str(self.total_num))
            print(self.total_num)
            self.total_num = 0
        except Exception as e:
            print(e)

    def thread_work(self):
        thread = threading.Thread(target=self.work, args=())
        if not thread.is_alive():
            thread.start()

    # 1 裁切
    def __cropImage(self, img, boxes):
        """
        裁切
        :param img: 图像
        :param bboxes: 该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
        :return: crop_img：裁剪后的图像；crop_bboxes：裁剪后的boundingbox的坐标，list
        """
        # 裁剪图像
        w = img.shape[1]
        h = img.shape[0]

        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        # 最小区域
        for bbox in boxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
            name = bbox[4]

        # 包含所有目标框的最小框到各个边的距离
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # 随机扩展这个最小范围
        crop_x_min = int(x_min - uniform(0, d_to_left))
        crop_y_min = int(y_min - uniform(0, d_to_top))
        crop_x_max = int(x_max + uniform(0, d_to_right))
        crop_y_max = int(y_max + uniform(0, d_to_bottom))

        # 确保不出界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # 裁剪bounding boxes
        crop_bboxes = list()
        for bbox in boxes:
            crop_bboxes.append([int(bbox[0] - crop_x_min), int(bbox[1] - crop_y_min),
                                int(bbox[2] - crop_x_min), int(bbox[3] - crop_y_min), bbox[4]])

        return crop_img, crop_bboxes

    # 2-平移
    def __translationImage(self, img, boxes):
        """
        平移
        :param img: img
        :param bboxes: bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
        :return: shift_img：平移后的图像array；shift_bboxes：平移后的boundingbox的坐标，list
        """

        # 平移图像
        w = img.shape[1]
        h = img.shape[0]

        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        for bbox in boxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(x_max, bbox[3])
            name = bbox[4]

        # 包含所有目标框的最小框到各个边的距离，即每个方向的最大移动距离
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # 在矩阵第一行中表示的是[1,0,x],其中x表示图像将向左或向右移动的距离，如果x是正值，则表示向右移动，如果是负值的话，则表示向左移动。
        # 在矩阵第二行表示的是[0,1,y],其中y表示图像将向上或向下移动的距离，如果y是正值的话，则向下移动，如果是负值的话，则向上移动。
        x = uniform(-(d_to_left / 3), d_to_right / 3)
        y = uniform(-(d_to_top / 3), d_to_bottom / 3)
        M = float32([[1, 0, x], [0, 1, y]])

        # 仿射变换
        shift_img = warpAffine(img, M,
                               (img.shape[1], img.shape[0]))  # 第一个参数表示我们希望进行变换的图片，第二个参数是我们的平移矩阵，第三个希望展示的结果图片的大小

        # 平移boundingbox
        shift_bboxes = list()
        for bbox in boxes:
            shift_bboxes.append([int(bbox[0] + x), int(bbox[1] + y), int(bbox[2] + x), int(bbox[3] + y), bbox[4]])

        return shift_img, shift_bboxes

    # 3-改变亮度
    def __changeLightofImage(self, img, boxes):
        """
        改变亮度
        :param img: 图像
        :return: img：改变亮度后的图像array
        """
        '''
        adjust_gamma(image, gamma=1, gain=1)函数:
        gamma>1时，输出图像变暗，小于1时，输出图像变亮
        '''
        flag = uniform(self.light_down, self.light_up)  ##flag>1为调暗,小于1为调亮
        newBoxes = deepcopy(boxes)
        newImage = exposure.adjust_gamma(img, flag)
        return newImage, newBoxes

    # 4-添加高斯噪声
    def __addNoiseToImage(self, img, boxes):
        """
        加入噪声
        :param img: 图像
        :return: img：加入噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """
        newBoxes = deepcopy(boxes)
        newImage = random_noise(img, mode='gaussian', clip=True) * 255
        return newImage, newBoxes

    # 5-旋转
    def __rotateImage(self, img, boxes):
        """
        旋转
        :param img: 图像
        :param boxes:
        :param angle: 旋转角度
        :param scale: 默认1
        :return: rot_img：旋转后的图像array；rot_bboxes：旋转后的boundingbox坐标list
        """
        '''
        输入:
            img:array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:
            scale:默认1
        输出:

        '''
        # 旋转图像
        w = img.shape[1]
        h = img.shape[0]
        angle = uniform(self.rotate_down, self.rotate_up)
        scale = uniform(0.5, 1.5)
        # 角度变弧度
        rangle = deg2rad(angle)
        # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
        nw = (abs(sin(rangle) * h) + abs(cos(rangle) * w)) * scale
        nh = (abs(cos(rangle) * h) + abs(sin(rangle) * w)) * scale
        # 获取图像绕着某一点的旋转矩阵
        # getRotationMatrix2D(Point2f center, double angle, double scale)
        # Point2f center：表示旋转的中心点
        # double angle：表示旋转的角度
        # double scale：图像缩放因子
        rot_mat = getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
        # 新中心点与旧中心点之间的位置
        rot_move = dot(rot_mat, array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = warpAffine(img, rot_mat, (int(ceil(nw)), int(ceil(nh))),
                             flags=INTER_LANCZOS4)  # ceil向上取整

        # 矫正boundingbox
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in boxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            name = bbox[4]
            point1 = dot(rot_mat, array([(x_min + x_max) / 2, y_min, 1]))
            point2 = dot(rot_mat, array([x_max, (y_min + y_max) / 2, 1]))
            point3 = dot(rot_mat, array([(x_min + x_max) / 2, y_max, 1]))
            point4 = dot(rot_mat, array([x_min, (y_min + y_max) / 2, 1]))

            # 合并np.array
            concat = vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
            # 改变array类型
            concat = concat.astype(int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, name])
        return rot_img, rot_bboxes

    # 6-镜像
    def __flipImage(self, img, bboxes, val):
        """
        镜像
        :param self:
        :param img:
        :param bboxes:
        :return:
        """
        '''
        镜像后的图片要包含所有的框
        输入：
            img：图像array
            bboxes：该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            flip_img:镜像后的图像array
            flip_bboxes:镜像后的bounding box的坐标list
        '''
        # 镜像图像
        import copy
        flip_img = deepcopy(img)
        if val == -1:
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转
            flip_img = flip(flip_img, -1)
        else:
            flip_img = flip(flip_img, 0)
        # ---------------------- 矫正boundingbox ----------------------
        flip_bboxes = list()
        for bbox in bboxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            name = bbox[4]
            if horizon:
                flip_bboxes.append([w - x_max, h - y_min, w - x_min, h - y_max, name])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min, name])

        return flip_img, flip_bboxes


if __name__ == '__main__':
    app = QApplication(argv)
    app.setStyleSheet(StyleSheet)
    win = Winform()
    win.show()
    exit(app.exec_())

