import numpy as np
import torch
from Dataloader.S3DIS import get_sets
from utils.test_perform_cal import get_mean_accuracy
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from utils.cal_final_result import accuracy_calculation
from model.pointnet import PointNetDenseCls
import argparse


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())




def get_parse():
    parser=argparse.ArgumentParser(description='argumment')
    parser.add_argument('--novel',type=bool)
    parser.add_argument('--datapath',type=str,default='/data1/jiajing/dataset/plane_seg_sample')
    parser.add_argument('--cuda',type=int)
    parser.add_argument('--exp_name',type=str)

    return parser.parse_args()







def main(train,novel,plane_seg,exp_name):
    seed=1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
    
    torch.backends.cudnn.enabled=False
    inpt_length=11 if novel else 9
    
    cuda=0
    datapath='D:/Computer_vision/3D_Dataset/Stanford_Large_Scale/plane_seg_sample'
    #set seed 
    # torch.random.seed()
    # model=get_model(13,inpt_length=inpt_length)
    if plane_seg:
        k=2
    else: k=13
    
    model=PointNetDenseCls(k=k,inpt_length=inpt_length)
    
    train_loader,test_loader,valid_loader=get_sets(datapath,batch_size=25,novel=novel,plane_seg=plane_seg)
    
    if train:
        train_model(model,train_loader,valid_loader,exp_name,cuda)
    
    if not train:
        test(model,test_loader,pth_folder=exp_name)
        
def test(model,data_loader,pth_folder):
    pth_file=os.path.join('./pth_file',pth_folder)
    pth_list=os.listdir(pth_file)
    index_list=np.array([int(i.split('_')[-1]) for i in pth_list])
    target_pth=pth_list[np.argmax(index_list)]
    target_pth_path=os.path.join(pth_file,target_pth)

    dic=torch.load(target_pth_path)
    model.eval()
    model.load_state_dict(dic['model_state'])
    device=next(model.parameters()).device

    num_cls=13
    confusion_matrix=np.zeros((num_cls,num_cls))
    tq_it=tqdm(data_loader,ncols=100,unit='batch')

    for indice,(x_cpu,y_cpu) in enumerate(tq_it):
        x=x_cpu.to(device)
        out=model(x)[0]
        out=out.cpu().detach().numpy()
        out=out.reshape(-1,num_cls)
        out=np.argmax(out,axis=1).reshape(-1)
        label=y_cpu.view(-1).numpy()
        for i in range(out.shape[0]):
            truth=label[i]
            prediction=out[i]
            confusion_matrix[truth,prediction]+=1
    
    calcula_class=accuracy_calculation(confusion_matrix)
    np.save('./result/{}'.format(pth_folder),calcula_class)
    print(calcula_class.get_over_all_accuracy())
    print(calcula_class.get_intersection_union_per_class())
    print(calcula_class.get_average_intersection_union())







def train_model(model,train_loader,valid_loader,exp_name,cuda_n):
    assert torch.cuda.is_available()
    
    #这里应该用GPU
    device=torch.device('cuda:{}'.format(cuda_n))
    model=model.to(device)
    
    # device=torch.device('cpu')
    # model=model.to(device)
    initial_epoch=0
    training_epoch=80

    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    # lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                             milestones=np.arange(10,training_epoch,20),gamma=0.7)





    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        
        #真正训练这里应该解封
        epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary


    def eval_one_epoch():
        iteration=tqdm(valid_loader,ncols=100,unit='batch',leave=False)
        #epsum only have logit and labes
        #epsum['logti'] is (batch,4096,13)
        #epsum['labels] is (batch,4096)
        
        epsum=run_one_epoch(model,iteration,"valid")
        mean_acc=np.mean(epsum['acc'])
        summary={'meac':mean_acc}
        return summary



    #build tensorboard
    
    tensorboard=SummaryWriter(log_dir='./TB/{}'.format(exp_name))
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)

    #build folder for pth_file
    pth_save_path=os.path.join('./pth_file',exp_name)
    if not os.path.exists(pth_save_path):
        os.mkdir(pth_save_path)
    
    for e in tqdm_epoch:
        train_summary=train_one_epoch()
        valid_summary=eval_one_epoch()
        summary={**train_summary,**valid_summary}
        # lr_schedule.step()
        #save checkpoint
        if (e%5==0) or (e==training_epoch-1):
            summary_saved={**train_summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict()}

            torch.save(summary_saved,'./pth_file/{0}/epoch_{1}'.format(exp_name,e))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    


def run_one_epoch(model,tqdm_iter,mode,loss_func=None,optimizer=None,loss_interval=10):
    if mode=='train':
        model.train()
    else:
        model.eval()
        param_grads=[]
        for param in model.parameters():
            param_grads+=[param.requires_grad]
            param.requires_grad=False
    
    summary={"losses":[],"acc":[]}
    device=next(model.parameters()).device

    for i,(x_cpu,y_cpu) in enumerate(tqdm_iter):
        x,y=x_cpu.to(device),y_cpu.to(device)

        if mode=='train':
            optimizer.zero_grad()
        
        logits=model(x)
        if loss_func is not None:
            loss=loss_func(logits.reshape(-1,logits.shape[-1]),y.view(-1))
            summary['losses']+=[loss.item()]
        
        if mode=='train':
            loss.backward()
            optimizer.step()

            #display
            if loss_func is not None and i%loss_interval==0:
                tqdm_iter.set_description("Loss: %.3f"%(np.mean(summary['losses'])))

        else:
            log=logits.cpu().detach().numpy()
            lab=y_cpu.numpy()
            num_cls=model.k
            
            mean_acc=get_mean_accuracy(log,lab,num_cls)
            summary['acc'].append(mean_acc)

        
        # summary['logits']+=[logits.cpu().detach().numpy()]
        # summary['labels']+=[y_cpu.numpy()]
            if i%loss_interval==0:
                tqdm_iter.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))


    if mode!='train':
        for param,value in zip(model.parameters(),param_grads):
                param.requires_grad=value

    # summary["logits"] = np.concatenate(summary["logits"], axis=0)
    # summary["labels"] = np.concatenate(summary["labels"], axis=0)

    return summary


if __name__=='__main__':
    # parser=get_parse()
    # data_path='/data1/jiajing/dataset/plane_seg_sample'
    main(train=True,novel=False,plane_seg=True,exp_name='PointNet_plane_136')
