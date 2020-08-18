import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data
import os
np.set_printoptions(suppress=True)


# cls_list = ['clutter', 'ceiling', 'floor', 
# 'wall', 'beam', 'column', 'door','window', 
# 'table', 'chair', 'sofa', 'bookcase', 'board']


np.random.seed(0)
class S3DISDataset(data.Dataset):
    def __init__(self,root,split,novel,plane_seg):
        if split=='train':
            self.area_list=['Area_1','Area_2','Area_3','Area_4','Area_6']
        else:
            self.area_list=['Area_5']
        self.plane_seg=plane_seg
        self.plane_cls_list=[1,2,3]
        
        self.novel=novel
        self.root=root
        self.batch_list=self.create_batch_list()
       
        
    def create_batch_list(self):
        all_batch_list=[]
        for area in self.area_list:
            area_path=os.path.join(self.root,area)
            room_list=os.listdir(area_path)
            for room in room_list:
               batch_folder_path=os.path.join(area_path,room,'plane_seg_batch')
               batch_list=os.listdir(batch_folder_path)
               for batch in batch_list:
                   batch_path=os.path.join(batch_folder_path,batch)
                   all_batch_list.append(batch_path)
        
        return all_batch_list
    
    def __getitem__(self,batch_index):
        txt_file=self.batch_list[batch_index]
        data=np.loadtxt(txt_file)
        if self.novel:
            inpt=torch.FloatTensor(data[:,0:11])
        else:
            inpt=torch.FloatTensor(data[:,0:9])
        label=torch.LongTensor(data[:,-1])
        
        if self.plane_seg:
            label_plane=torch.zeros_like(label)
            for i in self.plane_cls_list:
                label_plane[label==i]=1
            label=label_plane
        
        return inpt,label

    def __len__(self):
        return 30
        # return len(self.batch_list)

    
    # def visulize_point(self,room_index):
    #     txt_file=self.batch_list[room_index]
    #     data=np.loadtxt(txt_file)
        
    #     #point info
    #     points_info=o3d.geometry.PointCloud()
    #     points_info.points=o3d.utility.Vector3dVector(data[:,0:3])
    #     points_info.colors=o3d.utility.Vector3dVector(data[:,3:6]/255)
    #     o3d.visualization.draw_geometries([points_info])
    
    
    

def get_sets(data_path,batch_size,novel,plane_seg):
    train_data=S3DISDataset(data_path,split='train',novel=novel,plane_seg=plane_seg)
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=False,num_workers=2)

    test_data=S3DISDataset(data_path,split='test',novel=novel,plane_seg=plane_seg)
    test_loader=data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=2)

    valid_loader=S3DISDataset(data_path,split='valid',novel=novel,plane_seg=plane_seg)
    valid_loader=data.DataLoader(dataset=valid_loader,batch_size=batch_size,shuffle=False,num_workers=2)
    
    return train_loader,test_loader,valid_loader





def visulize_point(point_path,cls=0):
    data=np.load(point_path)
    if cls!=None:
        pos=(data[:,-2]==cls)
        data[pos,3:6]=np.array([255,0,0])
    
    
    points_info=o3d.geometry.PointCloud()
    points_info.points=o3d.utility.Vector3dVector(data[:,0:3])
    points_info.colors=o3d.utility.Vector3dVector(data[:,3:6]/255)
    o3d.visualization.draw_geometries([points_info])



if __name__=='__main__':
    data_path='D:/Computer_vision/3D_Dataset/Stanford_Large_Scale/plane_seg_sample'
    dataset=S3DISDataset(data_path,split='train',novel=False,plane_seg=True)
    inpt,label=dataset[20]
    # point_path='D:/Computer_vision/3D_Dataset\Stanford_Large_Scale/plane_seg_sample/Area_1/conferenceRoom_1/whole_room_point.npy'
    # visulize_point(point_path)
    
