# -*- coding: utf-8 -*-
# @Author     : weiz
# @Time       : 2020/8/4 下午3:01
# @FileName   : dataPrecess.py
# @Software   : PyCharm
# @Description: 处理PASCAL VOC相关的数据
import os
from xml.dom.minidom import parse
import xml.etree.ElementTree as ET
import random
import cv2

def changeXMLImagePath(xmlPath, savePath):
    """
    修改xml文件中图片的路径
    :param xmlPath:
    :param savePath:
    :return:
    """
    xmlNames = os.listdir(xmlPath)
    for xmlName in xmlNames:
        xmlPath_tmp = os.path.join(xmlPath, xmlName)

        domTree = parse(xmlPath_tmp)        # 读取xml文件
        rootNode = domTree.documentElement  # 文档根元素
        pathNode = rootNode.getElementsByTagName("path")[0]
        pathNode.childNodes[0].data = pathNode.childNodes[0].data.split('\\')[-1]  # 只取图片的名字

        savePath_tmp = os.path.join(savePath, xmlName)
        with open(savePath_tmp, 'w') as f:
            domTree.writexml(f, addindent='  ', encoding='utf-8')  # 缩进 - 换行 - 编码


def checkXMLAndImage(xmlPath, imagePath, isDelete=None):
    """
    检测xml文件和图片是否一一对应
    :param xmlPath:
    :param imagePath:
    :param isDelete:
    :return:
    """
    if (isDelete == None) or (isDelete != True):
        isDelete = False


    xmlNames = os.listdir(xmlPath)
    imageNames = os.listdir(imagePath)

    for i in range(len(xmlNames)):
        xmlNames[i] = xmlNames[i].split('.')[0]

    for i in range(len(imageNames)):
        imageNames[i] = imageNames[i].split('.')[0]

    xmlSet = set(xmlNames)
    imageSet = set(imageNames)
    intersection = xmlSet & imageSet

    xmls_diff = list(xmlSet - intersection)
    images_diff = list(imageSet - intersection)
    print("xml文件是否多余:", xmls_diff)
    print("图片是否多余:", images_diff)
    if isDelete:
        for xml in xmls_diff:
            xmlPath_tmp = os.path.join(xmlPath, xml+".xml")
            os.remove(xmlPath_tmp)

        for image in images_diff:
            # 如果图片不是png格式时候，这里会有问题
            imagePath_tmp = os.path.join(imagePath, image+".png")
            os.remove(imagePath_tmp)

def fractionalSet(imagePath, txtSavePath):
    """
    划分测试集，验证集，训练集
    :param imagePath:
    :param txtSavePath:
    :return:
    """
    # 总数据：测试集+验证集+训练集；其中trainValPercent + trainPercent = 1
    testPercent = 0.0
    trainValPercent = 0.1
    trainPercent = 0.9

    imageNames = os.listdir(imagePath)
    testNum = int(len(imageNames) * testPercent)            # 测试集的数量
    trainValNum = len(imageNames) - testNum                 # 训练集+验证集的数量
    valNum = int(trainValNum * trainValPercent)             # 验证集的数量
    trainNum = trainValNum - valNum - testNum               # 训练集的数量

    test = random.sample(imageNames, testNum)
    imageNamesTrainVal = list(set(imageNames) - set(test))
    val = random.sample(imageNamesTrainVal, valNum)
    train = list(set(imageNamesTrainVal) - set(val))

    ftrainval_path = os.path.join(txtSavePath, "trainval.txt")   # 训练集+验证集
    ftest_path = os.path.join(txtSavePath, "test.txt")           # 测试集
    ftrain_path = os.path.join(txtSavePath, "train.txt")         # 训练集
    fval_path = os.path.join(txtSavePath, "val.txt")             # 验证集

    ftrainval = open(ftrainval_path, 'w')
    ftest = open(ftest_path, 'w')
    ftrain = open(ftrain_path, 'w')
    fval = open(fval_path, 'w')

    for imageName in imageNames:
        if imageName in test:
            imageName = imageName.split('.')[0] + "\n"
            ftest.write(imageName)
        else:
            if imageName in val:
                imageName = imageName.split('.')[0] + "\n"
                fval.write(imageName)
            else:
                imageName = imageName.split('.')[0] + "\n"
                ftrain.write(imageName)
            ftrainval.write(imageName)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def convert(size, box):
    """
    返回归一化后的中心点，宽高值
    :param size:
    :param box:
    :return:
    """
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(xmlPath, classes, txtSavePath):
    """
    将数据转成darknet可以训练的格式
    :param ret: 返回的结果
    :param xmlPath:
    :param classes:
    :param txtSavePath:
    :return:
    """
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    txtFileRet = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if int(difficult) == 1:
            continue

        try:
            cls_id = classes.index(cls)
        except ValueError:
            print("The category does not exist in the myData.names file!")
            exit()
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        txtFileRet.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    txtFile = open(txtSavePath, 'w')
    for line in txtFileRet:
        txtFile.write(line)
    txtFile.close()

def darknetTrainFormatData(xmlPath, imageSets, savePath, classes, sets):
    """
    生成darknet可以训练的数据格式
    :param xmlPath: xml文件所在的路径
    :param savePath: 修改后所保存的路径
    :return:
    """
    workRoot = os.getcwd()
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for setName in sets:
        filePath = os.path.join(imageSets, setName)         # 格式：./ImageSets/Main/test.txt
        imageNames = open(filePath).read().strip().split()  # 格式:['04_000059', '02_000120']

        setData = []
        for imageName in imageNames:
            setData.append(os.path.join(workRoot, "JPEGImages", imageName+".png"))
            xmlPath_img = os.path.join(xmlPath, imageName.split('.')[0]+".xml")   # 格式：./Annotations/15_000057.xml
            txtSavePath = os.path.join(savePath, imageName.split('.')[0]+".txt")  # 格式：./labels/15_000057.txt
            convert_annotation(xmlPath_img, classes, txtSavePath)

        setDataFile = open(setName, 'w')
        for line in setData:
            setDataFile.write(line+"\n")
        setDataFile.close()


def checkLabel(imagePath, labelPath):
    """
    随机抽查检查标注数据是否正确
    :param imagePath:
    :param labelPath:
    :return:
    """
    labelNames = os.listdir(labelPath)
    while True:
        index = random.randint(0, len(labelNames) - 1)
        labelPath_tmp = os.path.join(labelPath, labelNames[index])

        tree = ET.parse(labelPath_tmp)
        root = tree.getroot()
        imageName = root.find('filename').text
        labels = []   # 标签信息[[x,y,w,h,label]]
        for obj in root.iter('object'):
            x1 = int(obj.find("bndbox").find("xmin").text)
            y1 = int(obj.find("bndbox").find("ymin").text)
            x2 = int(obj.find("bndbox").find("xmax").text)
            y2 = int(obj.find("bndbox").find("ymax").text)
            name = obj.find("name").text
            labels.append([x1, y1, x2, y2, name])

        imagePath_tmp = os.path.join(imagePath, imageName)
        img = cv2.imread(imagePath_tmp)
        for label in labels:
            cv2.rectangle(img, (label[0], label[1]), (label[2], label[3]), (0, 0, 255))
            cv2.putText(img, label[-1], (label[0], label[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow(imageName, img)
        k = cv2.waitKey(0)

        # 按Esc键退出
        if k == 27:
            break
        # 按右箭头和下箭头继续检测
        if k != 83 and k != 84:
            if k == 27:
                break
            k = cv2.waitKey(0)
        if k == 27:
            break
        cv2.destroyAllWindows()


def readLineTXT(path):
    """
    按行读取txt文件
    :param path:
    :return:
    """
    txtLines = []
    file = open(path)
    for line in file:
        line = line.strip('\n')
        txtLines.append(line)
    file.close()
    return txtLines


def configPara():
    """
    配置yolo相关参数，设置classes的个数，以及filters参数等
    :return:
    """
    currPath = os.getcwd()
    dataPath = os.path.join(currPath, "cfg", "myData.data")
    namesPath = os.path.join(currPath, "cfg", "myData.names")
    cfgPath = os.path.join(currPath, "cfg", "myYolov4.cfg")

    # 读取myData.names
    namesList = readLineTXT(namesPath)

    # 设置myData.data
    dataList = []
    dataList.append("classes = " + str(len(namesList)))
    dataList.append("train = " + currPath + "/train.txt")
    dataList.append("valid = " + currPath + "/val.txt")
    dataList.append("names = " + namesPath)
    dataList.append("backup = " + currPath + "/weights")
    dataFile = open(dataPath, mode='w')
    for Line in dataList:
        dataFile.write(Line + '\n')
    dataFile.close()

    # 设置myYolov3.cfg
    oldCfgList = readLineTXT(cfgPath)
    filters_index = []
    isYolo = False
    for index, Line in enumerate(oldCfgList[::-1]):  # 找到需要修改的filters位置索引
        if "yolo" in Line:
            isYolo = True
        if ("filters" in Line) and (isYolo):
            filters_index.append(len(oldCfgList) - index - 1)
            isYolo = False
    newCfgList = []
    for index, line in enumerate(oldCfgList):        # 设置classes
        if "classes" in line:
            newCfgList.append("classes=" + str(len(namesList)))
        elif index in filters_index:                 # 设置filters
            newCfgList.append("filters="+str(3*(5+len(namesList))))
        else:
            newCfgList.append(line)
    cfgFile = open(cfgPath, mode='w')
    for index, Line in enumerate(newCfgList):
        if index == 0:
            cfgFile.write(Line)
        else:
            cfgFile.write('\n' + Line)
    cfgFile.close()


annotationsPath = "./Annotations"                # xml标注数据的路径
saveChangePath = "./Annotations"                 # xml修改后保存的路径
imageSets = "./ImageSets/Main"                   # 数据划分后保存的路径
jpegImages = "./JPEGImages"                      # 真实图片存放的路径
darknetFormatPath = "./labels"                   # 改成darknet可以训练数据后保存路径
classes = readLineTXT("./cfg/myData.names")      # 真实标签, 不能有空行
# 数据文件名，最好不要更改；如果更改，那么fractionalSet和configPara相应的也要更改
setNames = ["test.txt", "val.txt", "train.txt", "trainval.txt"]

# 注:1.运行的根目录是myData，下级目录有Annotations，ImageSets/Main，JPEGImages
#    2.处理的图片只能是一种图片的格式；这里只能处理png格式
if __name__ == "__main__":
    #checkLabel(jpegImages, annotationsPath)
    
    changeXMLImagePath(annotationsPath, saveChangePath)
    checkXMLAndImage(annotationsPath, jpegImages, True)
    fractionalSet(jpegImages, imageSets)

    darknetTrainFormatData(annotationsPath, imageSets, darknetFormatPath, classes, setNames)

    configPara()

