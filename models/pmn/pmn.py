import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN
import numpy as np

def adj_construction(inds, keypoint_num=17, symmetric=True):
	# # A = A+I
	# A = A + torch.eye(A.size(0))
	if inds:
		A = torch.zeros((keypoint_num, keypoint_num))
		for i, j in inds.items():
			A[i,j] = 1
	else:
		A = torch.ones((keypoint_num,keypoint_num))
	# D
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)

class PGception_Layer(nn.Module):	

	def __init__(self, in_channel, out_channel_list, branch_list=[0, 3], bias=True, drop=None, bn=False, agg_first=True, attn=False, init='kaiming_uniform'): # init = 'default', 'kaiming_uniform', 'xavier_uniform'
		super(PGception_Layer, self).__init__()
		# prepare adjacent matrixs
		keypoint_num = 17
		adj0 = torch.eye(keypoint_num)
		adj_all = adj_construction(inds=None)

		A = [adj0, adj_all]
		
		# import ipdb; ipdb.set_trace()
		self.branch_list = branch_list
		if 0 in self.branch_list:
			self.branch_0 = GCN(A[0], in_channel, out_channel_list[0], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first)
		if 3 in self.branch_list:
			self.branch_all = GCN(A[1], in_channel, out_channel_list[3], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first, attn=attn)
	def forward(self, x1, x2):
		# import ipdb; ipdb.set_trace()
		output = []
		if 0 in self.branch_list:
			branch_0 = self.branch_0(x2)
			output.append(branch_0)
		if 3 in self.branch_list:
			# import ipdb; ipdb.set_trace()
			branch_all = self.branch_all(x1)
			output.append(branch_all)
		
		return torch.cat(output, 2)

class Block(nn.Module):
	def __init__(self, in_channel, mid_channel, out_channel_list, branch_list, bias=True, drop=None, bn=False, agg_first=True, attn=False):
		super(Block, self).__init__()
		self.linear1 = nn.Linear(in_channel, mid_channel, bias)
		self.linear2 = nn.Linear(in_channel, mid_channel, bias)
		self.pgception = PGception_Layer(mid_channel, out_channel_list, branch_list, bias=bias, drop=drop, bn=bn, agg_first=agg_first, attn=attn)
		self.drop = drop
		self.bn = bn
		if drop:
			self.dropout = nn.Dropout(drop)
		if bn:
			self.batchnorm1 = nn.BatchNorm1d(17)
			self.batchnorm2 = nn.BatchNorm1d(17)

	def forward(self, x1, x2):
		# import ipdb; ipdb.set_trace()
		x1 = self.linear1(x1)
		x2 = self.linear2(x2)
		if self.bn:
			x1 = self.batchnorm1(x1)
			x2 = self.batchnorm2(x2)
		x1 = F.relu(x1)
		x2 = F.relu(x2)
		if self.drop:
			x1 = self.dropout(x1)
			x2 = self.dropout(x2)
		return self.pgception(x1, x2)

class PGception(nn.Module):
	def __init__(self, action_num=24, layers=1, classifier_mod="cat", o_c_l=[64,64], b_l=[0,3] ,last_h_c=256, bias=True, drop=None, bn=False, agg_first=True, attn=False):
		super(PGception, self).__init__()
		self.out_channel_list = np.array(o_c_l)
		self.branch_list = b_l
		self.classifier_mod = classifier_mod
		self.drop = drop
		self.bn = bn
		self.layers = layers
		# import ipdb; ipdb.set_trace()
		self.block1 = Block(in_channel= 2, mid_channel=128, out_channel_list=self.out_channel_list, branch_list=self.branch_list, bias=bias, drop=drop, bn=bn, agg_first=agg_first, attn=attn)
		if classifier_mod == "cat":
			# add a MLP to reduce the size of channels
			self.linear = nn.Linear(sum(self.out_channel_list[self.branch_list]), 64, bias=True)
			if drop:
				self.dropout = nn.Dropout(drop)
			if bn:
				self.batchnorm = nn.BatchNorm1d(17)
			self.classifier = nn.Sequential(
								nn.Linear(64*17, last_h_c, bias),
								nn.BatchNorm1d(last_h_c),
								nn.ReLU(inplace=True),
								nn.Dropout(drop),
								nn.Linear(last_h_c, action_num, bias),
							)

	def forward(self, x1, x2):
		'''
			x1 is the pose_to_box features,
			x2 is the pose_to_obj_offset features
		'''
		x = self.block1(x1, x2)
		if self.classifier_mod == "cat":
			x = self.linear(x)
			if self.bn:
				x = self.batchnorm(x)
			x = F.relu(x)
			if self.drop:
				x = self.dropout(x)

			return self.classifier(x.view(x.shape[0],-1))
        
if __name__ == "__main__":
	model = PGception()
	print(model)