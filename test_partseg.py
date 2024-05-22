"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score   
from sklearn.metrics import precision_score  
from sklearn.metrics import recall_score       
from sklearn.metrics import f1_score           # F1
from sklearn.metrics import mean_absolute_error # numpy MAE（Mean Absolute Error：）
from sklearn.metrics import mean_squared_error  # numpy MSE（Mean Square Error：）

"""
--model pointnet2_part_seg_msg 
--normal 
--log_dir pointnet2_part_seg_msg
"""

input_npoint = 40960
root = '../mydata/data_40960_7/'
TxtLine_trainacc= './trainacc.txt'
TxtLine_trainloss= './trainloss.txt'
num_classes = 1
num_part = 3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_classes = {'Airplane': [0, 1, 2]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}


###########################
# boll
arr_iou_boll = []
arr_acc_boll = []
arr_preci_boll = []
arr_recall_boll = []
arr_F1_boll = []
arr_mae_boll = []
arr_mse_boll = []

#leaf
arr_iou_leaf = []
arr_acc_leaf = []
arr_preci_leaf = []
arr_recall_leaf = []
arr_F1_leaf = []
arr_mae_leaf = []
arr_mse_leaf = []

#bar
arr_iou_stem = []
arr_acc_stem = []
arr_preci_stem = []
arr_recall_stem = []
arr_F1_stem = []
arr_mae_stem = []
arr_mse_stem = []

#all
arr_acc = []
arr_preci = []
arr_recall = []
arr_F1 = []
arr_mae = []
arr_mse = []




for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=input_npoint, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')
    return parser.parse_args()

def saveFPpoint(epoch, name, data):
    savepath = './output_test/'

    Newdata = data.cpu().detach().numpy()
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")

def getmin(data_one,minnum,data_two):
    data1 = data_one.copy()
    data2 = data_two.copy()
    pos = np.where(data1 <= minnum)  
    return data1[pos],data2[pos]

def getmiddle(data_one,minnum,maxnum,data_two):
    data1 = data_one.copy()
    data2 = data_two.copy()
    idx1 = np.where(data1 <= minnum )  #
    idx2 = np.where(data1 >= maxnum )  #
    data1[idx1[0]] = -1
    data1[idx2[0]] = -1
    
 
    pos = np.where(data1 >= 0)  #
    return data1[pos],data2[pos]

def getmax(data_one,maxnum,data_two):
    data1 = data_one.copy()
    data2 = data_two.copy()
    pos = np.where(data1 >= maxnum)  # 
    return data1[pos],data2[pos]



def  caluvalue_boll(target_squ,pred_squ):
  
    end_acc = accuracy_score(target_squ, pred_squ)
    # end_acc = (np.sum(target_squ == pred_squ))/len(pred_squ)
    end_preci2 = precision_score(target_squ, pred_squ, average='macro') 
    end_recall2 = recall_score(target_squ, pred_squ, average='macro')  
    end_F1 = f1_score(target_squ, pred_squ, average='weighted')  
    end_mae = mean_absolute_error(target_squ, pred_squ)
    end_mse = mean_squared_error(target_squ, pred_squ)

    arr_acc_boll.append(end_acc)
    arr_preci_boll.append(end_preci2)
    arr_recall_boll.append(end_recall2)
    arr_F1_boll.append(end_F1)
    arr_mae_boll.append(end_mae)
    arr_mse_boll.append(end_mse)
    #############################


def  caluvalue_leaf(target_squ,pred_squ):

    end_acc = accuracy_score(target_squ, pred_squ)
    end_preci2 = precision_score(target_squ, pred_squ, average='macro')  
    end_recall2 = recall_score(target_squ, pred_squ, average='macro')
    end_F1 = f1_score(target_squ, pred_squ, average='weighted')  
    end_mae = mean_absolute_error(target_squ, pred_squ)
    end_mse = mean_squared_error(target_squ, pred_squ)

    arr_acc_leaf.append(end_acc)
    arr_preci_leaf.append(end_preci2)
    arr_recall_leaf.append(end_recall2)
    arr_F1_leaf.append(end_F1)
    arr_mae_leaf.append(end_mae)
    arr_mse_leaf.append(end_mse)
    #############################



def  caluvalue_stem(target_squ,pred_squ):

    end_acc = accuracy_score(target_squ, pred_squ)
    end_preci2 = precision_score(target_squ, pred_squ, average='macro')  
    end_recall2 = recall_score(target_squ, pred_squ, average='macro')  
    end_F1 = f1_score(target_squ, pred_squ, average='weighted')  
    end_mae = mean_absolute_error(target_squ, pred_squ)
    end_mse = mean_squared_error(target_squ, pred_squ)

    arr_acc_stem.append(end_acc)
    arr_preci_stem.append(end_preci2)
    arr_recall_stem.append(end_recall2)
    arr_F1_stem.append(end_F1)
    arr_mae_stem.append(end_mae)
    arr_mse_stem.append(end_mse)
    #############################


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    TEST_DATASET = PartNormalDataset(root = root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" %  len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():



        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)
            #############################
            target_squ = target.squeeze(0)
            pred_squ = cur_pred_val.squeeze(0)
            #############################
            label0gt, label0pre = getmin(target_squ, 0.2, pred_squ)  
            label1gt, label1pre = getmiddle(target_squ, 0.2, 1.8, pred_squ)  
            label2gt, label2pre = getmax(target_squ, 1.8, pred_squ)  

            # boll 2
            caluvalue_boll(label2gt, label2pre)
            # leaf 1
            caluvalue_leaf(label1gt, label1pre)
            # stem
            caluvalue_stem(label0gt, label0pre)

            #############################
            # all
            end_acc = accuracy_score(target_squ, pred_squ)
            end_preci2 = precision_score(target_squ, pred_squ, average='macro')  
            end_recall2 = recall_score(target_squ, pred_squ, average='macro')  
            end_F1 =      f1_score(target_squ, pred_squ, average='weighted')  
            end_mae = mean_absolute_error(target_squ, pred_squ)
            end_mse = mean_squared_error(target_squ, pred_squ)

            arr_acc.append(end_acc)
            arr_preci.append(end_preci2)
            arr_recall.append(end_recall2)
            arr_F1.append(end_F1)
            arr_mae.append(end_mae)
            arr_mse.append(end_mse)
            #########################################




            ###  SAVE   cur_pred_val == target
            points = points.squeeze(0)
            points = points[0:3,:]

            temppre = torch.cat([points,torch.tensor(cur_pred_val).cuda() ],0)
            tempgt = torch.cat([points, torch.tensor(target).cuda() ], 0)
            saveFPpoint( batch_id , '_input', points.transpose(1, 0))
            saveFPpoint( batch_id , '_pre', temppre.transpose(1, 0))
            saveFPpoint(batch_id, '_gt', tempgt.transpose(1, 0))


            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            segp = cur_pred_val[0, :]
            segl = target[0, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))

            ####  part_ious is a list ##############
            arr_iou_stem.append(part_ious[0])
            arr_iou_leaf.append(part_ious[1])
            arr_iou_boll.append(part_ious[2])

            shape_ious[cat].append(np.mean(part_ious))
            mean_shape_ious = np.mean(list(shape_ious.values()))  

   


        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))  
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np))
        for cat in sorted(shape_ious.keys()):
            print(len(cat))
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


    log_string('Accuracy is: %.5f'%test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f'%test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f'%test_metrics['inctance_avg_iou'])


    print('11111111   boll  1111111111111111111111')
    print("miou", sum(arr_iou_boll) / len(arr_iou_boll))
    print("acc", sum(arr_acc_boll) / len(arr_acc_boll))
    print("preci", sum(arr_preci_boll) / len(arr_preci_boll))
    print("recall", sum(arr_recall_boll) / len(arr_recall_boll))
    print("F1", sum(arr_F1_boll) / len(arr_F1_boll))
    print("mae", sum(arr_mae_boll) / len(arr_mae_boll))
    print("mse", sum(arr_mse_boll) / len(arr_mse_boll))

    print('11111111   leaf  1111111111111111111111')
    print("miou", sum(arr_iou_leaf) / len(arr_iou_leaf))
    print("acc", sum(arr_acc_leaf) / len(arr_acc_leaf))
    print("preci", sum(arr_preci_leaf) / len(arr_preci_leaf))
    print("recall", sum(arr_recall_leaf) / len(arr_recall_leaf))
    print("F1", sum(arr_F1_leaf) / len(arr_F1_leaf))
    print("mae", sum(arr_mae_leaf) / len(arr_mae_leaf))
    print("mse", sum(arr_mse_leaf) / len(arr_mse_leaf))

    print('11111111   stem 1111111111111111111111')
    print("miou", sum(arr_iou_stem) / len(arr_iou_stem))
    print("acc", sum(arr_acc_stem) / len(arr_acc_stem))
    print("preci", sum(arr_preci_stem) / len(arr_preci_stem))
    print("recall", sum(arr_recall_stem) / len(arr_recall_stem))
    print("F1", sum(arr_F1_stem) / len(arr_F1_stem))
    print("mae", sum(arr_mae_stem) / len(arr_mae_stem))
    print("mse", sum(arr_mse_stem) / len(arr_mse_stem))

    print('11111111  all  1111111111111111111111')

    print("acc", sum(arr_acc) / len(arr_acc))
    print("preci", sum(arr_preci) / len(arr_preci))
    print("recall", sum(arr_recall) / len(arr_recall))
    print("F1", sum(arr_F1) / len(arr_F1))
    print("mae", sum(arr_mae) / len(arr_mae))
    print("mse", sum(arr_mse) / len(arr_mse))


if __name__ == '__main__':
    args = parse_args()
    main(args)

