import torch
import subjectlist as subl
import os
import torchsrc

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

# hyper parameters
epoch_num = 101
# learning_rate = 0.00001
start_epoch = 0

network = 'UNet3D'

#experiments1
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment1'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize'
# learning_rate = 0.00001   #default 0.0001
# res = [168,168,64]
# imsize = [168,168,64]
# sample_size = 168
# sample_duration = 64
# batch_size = 1
# fcnum = 2048
# networkName ='resnet101'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# add_calcium_mask = False
# data_augmentation = False
# dual_network = False
# use_siamese = False

#experiments2
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment2'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_lungseg'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = 128
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='resnet101'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# add_calcium_mask = False
# data_augmentation = False
# dual_network = False
# use_siamese = False

# #experiment 3
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment3'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_lungseg'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# add_calcium_mask = Falsez
# data_augmentation = False
# dual_network = False
# use_siamese = False

#experiment 4
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment4'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_lungseg'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 2
# fcnum = 1536
# networkName ='densenetyh'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# add_calcium_mask = False
# data_augmentation = False
# dual_network = False
# use_siamese = False

#
# #experiment 5
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment5'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = False
# data_augmentation = False
# dual_network = False
# use_siamese = False

# #experiment 6
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment6'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless_calciumMask'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = False
# data_augmentation = True
# dual_network = False
# use_siamese = False

#
# #experiment 7
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment7'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_bone_and_heart'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless_calciumMask'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = False
# data_augmentation = True
# dual_network = False
# use_siamese = False

# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = False
# data_augmentation = True
# dual_network = True
# use_siamese = False


#experiment 9   #aug only rotate+translation, simple dual
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment9'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = False
# data_augmentation = True
# dual_network = True
# use_siamese = False

# #experiment 10   #aug only rotate
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment10'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = False

#
# #experiment 11   #aug only rotate
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment11'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = True

# # #experiment 12   #aug only rotate 0.01*sieame
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment12'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = True

# # #experiment 13   #aug only rotate  0.001*sieame
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment13'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.00001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = True
# use_attention = False

# # #experiment 14   #aug only rotate  0.001*sieame , test attention
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment14'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='sononet_grid_attention'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = False

#
# # #experiment 15  #aug only rotate  0.001*sieame , test attention
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment15'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_excludeUseless'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='sononet_grid_attention_v2'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = False


#experimentt 16
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment16'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 4
# fcnum = 1536
# networkName ='sononet_grid_attention'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = False


# # #experimentt 17
#exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment17'
#input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
#calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
#learning_rate = 0.0001   #default 0.0001
#res = [192,128,64]
#imsize = [192,128,64]
#sample_size = [128,128]
#sample_duration = 64
#batch_size = 2
#fcnum = 1536
#networkName ='huo_net'
#input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
#input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
#test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_201807011.csv'
#new_img_root_dir = []
#add_calcium_mask = True
#data_augmentation = True
#dual_network = True
#use_siamese = False
#ValidateAttention = False
#siamese_coeiff = 0.0001
#clss_num = 3


# #experimentt 18
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment18'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 2
# fcnum = 1536
# networkName =  'huo_net_conv1'  #'huo_net_direct' #''densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = False
# ValidateAttention = False
# siamese_coeiff = 0.0001
# clss_num = 3



# #experimentt 19
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment19'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 2
# fcnum = 1536
# networkName =  'huo_net_conv1'  #'huo_net_direct' #''densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = False
# ValidateAttention = False
# siamese_coeiff = 0.0001
# clss_num = 2


# experiment 20
exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment20'
input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
learning_rate = 0.0001   #default 0.0001
res = [192,128,64]
imsize = [192,128,64]
sample_size = [128,128]
sample_duration = 64
batch_size = 2
fcnum = 1536
networkName =  'huo_net_conv1'  #'huo_net_direct' #''densenet121_twochanel'
input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_2018_08_03.csv'
new_img_root_dir = []
add_calcium_mask = True
data_augmentation = True
dual_network = True
use_siamese = True
ValidateAttention = True
siamese_coeiff = 0.001
clss_num = 3


#experiment 21
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment21'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = [128,128]
# sample_duration = 64
# batch_size = 2
# fcnum = 1536
# networkName =  'huo_net_conv1'  #'huo_net_direct' #''densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_list_201807011.csv'
# input_demographic_CAC2_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC2_list_201807011.csv'
# test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_201807011.csv'
# new_img_root_dir = []
# add_calcium_mask = True
# data_augmentation = True
# dual_network = True
# use_siamese = True
# ValidateAttention = False
# siamese_coeiff = 0.01
# clss_num = 3



# experiments22
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment22'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = 128
# sample_duration = 64
# batch_size = 2
# fcnum = 1536
# networkName ='resnet101'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_and2_list_201807011.csv'
# input_demographic_CAC2_file = ' '
# test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_201807011.csv'
# add_calcium_mask = True
# data_augmentation = True
# dual_network = False
# use_siamese = False
# ValidateAttention = False
# siamese_coeiff = 0.0001
# clss_num = 3


# experiments23
# exp_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/Experiment23'
# input_img_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResampleNormalize_Final_img'
# calcium_mask_dir = '/fs4/masi/huoy1/JeffFHSCT/Experiments/DataResample_Final_calcium'
# learning_rate = 0.0001   #default 0.0001
# res = [192,128,64]
# imsize = [192,128,64]
# sample_size = 128
# sample_duration = 64
# batch_size = 2
# fcnum = 1536
# networkName ='densenet121_twochanel'
# input_demographic_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-11-2018_madeByYuankai/FH_Export_CAC1_and2_list_201807011.csv'
# input_demographic_CAC2_file = ' '
# test_calcium_CAC1_file = '/fs4/masi/huoy1/JeffFHSCT/demographic/07-23-2018_testAttentionYuankai/FH_Export_CAC1_list_201807011.csv'
# add_calcium_mask = True
# data_augmentation = True
# dual_network = False
# use_siamese = False
# ValidateAttention = False
# siamese_coeiff = 0.0001
# clss_num = 3











if ValidateAttention:
	batch_size = 1
	train_img_subs,train_img_files,train_cateegories,test_img_subs,test_img_files,test_cateegories = subl.get_demographic_file(test_calcium_CAC1_file)
else:
	train_img_subs,train_img_files,train_cateegories,test_img_subs,test_img_files,test_cateegories = subl.get_demographic_file(input_demographic_CAC1_file)

# define paths
working_dir = os.path.join(exp_dir, 'working_dir')


# make img list
train_dict = {}
train_dict['img_subs'] = train_img_subs
train_dict['img_files'] = train_img_files
train_dict['categories'] = train_cateegories
test_dict = {}
test_dict['img_subs'] = test_img_subs
test_dict['img_files'] = test_img_files
test_dict['categories'] = test_cateegories


# load image
if add_calcium_mask:
	train_set = torchsrc.imgloaders.pytorch_loader_clss3D_calcium(train_dict, num_labels=clss_num,
																  input_root_dir=input_img_dir,calcium_mask_dir=calcium_mask_dir,
																  res=res, imsize=imsize,dual_network = dual_network,
																  data_augmentation=data_augmentation)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
	test_set = torchsrc.imgloaders.pytorch_loader_clss3D_calcium(test_dict, num_labels=clss_num, input_root_dir=input_img_dir,calcium_mask_dir=calcium_mask_dir,
																 res=res,
																 imsize=imsize, data_augmentation=False)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

else:
	train_set = torchsrc.imgloaders.pytorch_loader_clss3D(train_dict, num_labels=clss_num, input_root_dir=input_img_dir,
														  res=res, imsize=imsize,dual_network = dual_network,data_augmentation=data_augmentation)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
	test_set = torchsrc.imgloaders.pytorch_loader_clss3D(test_dict, num_labels=clss_num, input_root_dir=input_img_dir, res=res,
														 imsize=imsize,data_augmentation=False)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

# train_set = torchsrc.imgloaders.pytorch_loader_no255(train_dict,num_labels=lmk_num)
# train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1)
# test_set = torchsrc.imgloaders.pytorch_loader_no255(test_dict,num_labels=lmk_num)
# test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=1)

# load network

resnet_shortcut = 'B'

if networkName =='resnet101':
	model = torchsrc.models.resnet101(num_classes=clss_num,
					shortcut_type=resnet_shortcut,
					sample_size=sample_size,
					sample_duration=sample_duration,
					fcnum = fcnum)

elif networkName =='densenet121':
	model = torchsrc.models.densenet121(sample_size0=sample_size[0],
				 sample_size1=sample_size[1],
				 sample_duration=sample_duration,
				)
elif networkName == 'densenet121_twochanel':
	model = torchsrc.models.densenet121_twochanel(
		sample_size0=sample_size,
		sample_size1=sample_size,
		sample_duration=sample_duration,
		# num_classes = clss_num
	)
elif networkName == 'densenet121_twochanel_cam':
	model = torchsrc.models.densenet121_twochanel_cam(sample_size0=sample_size[0],
		sample_size1=sample_size[1],
		sample_duration=sample_duration,
		# num_classes = clss_num
	)
elif networkName == 'sononet_grid_attention':
	model = torchsrc.models.sononet_grid_attention(
		nonlocal_mode='concatenation_mean_flow',
		aggregation_mode='concat',
		# num_classes = clss_num
	)
elif networkName == 'sononet_grid_attention_v2':
	model = torchsrc.models.sononet_grid_attention_v2(
		nonlocal_mode='concatenation_softmax',
		aggregation_mode='concat',
		# num_classes = clss_num
	)
elif networkName == 'densenetyh':
	model = torchsrc.models.densenetyh(sample_size0=sample_size[0],
		sample_size1=sample_size[1],
		sample_duration=sample_duration,
		# num_classes = clss_num
	)
elif networkName == 'huo_net':
	model = torchsrc.models.huo_net(
		nonlocal_mode='concatenation_softmax',
		aggregation_mode='concat',
		# num_classes = clss_num
	)
elif networkName == 'huo_net_direct':
	model = torchsrc.models.huo_net_direct(
		nonlocal_mode='concatenation_softmax',
		aggregation_mode='concat',
		# num_classes = clss_num
	)
elif networkName == 'huo_net_conv1':
	model = torchsrc.models.huo_net_conv1(
		nonlocal_mode='concatenation_softmax',
		aggregation_mode='concat',
		n_classes = clss_num
	)


out = os.path.join(working_dir, 'ResNet3D_out_0.00001')
mkdir(out)

# model = torchsrc.models.VNet()

print_network(model)
#
# load optimizor
optim = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
# optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# load CUDA
cuda = torch.cuda.is_available()
#cuda = False
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

# load trainer
trainer = torchsrc.Trainer(
	cuda=cuda,
	model=model,
	optimizer=optim,
	train_loader=train_loader,
	test_loader=test_loader,
	out=out,
	max_epoch = epoch_num,
	batch_size = batch_size,
	lmk_num = clss_num,
	dual_network = dual_network,
	add_calcium_mask = add_calcium_mask,
	use_siamese = use_siamese,
	siamese_coeiff = siamese_coeiff,
)


print("==start training==")


start_iteration = 1
trainer.epoch = start_epoch
if ValidateAttention:
	trainer.epoch = 84
	trainer.max_epoch = trainer.epoch+1
	trainer.test_epoch()
else:
	trainer.iteration = start_iteration
	trainer.train_epoch()









