import torchvision.models as models
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class video_feature_extraction(nn.Module):
	def __init__(self,model,transfer_dim):
		super(video_feature_extraction, self).__init__()
		
		if model=='resnet18':
			self.renet=models.resnet18(pretrained=True)
		self.fc=nn.Linear(1000,transfer_dim)

	def forward(self,x):
		x=self.resnet(x)
		x=self.fc(x)
		return x

if __name__=='__main__':
	img=mpimg.imread('./1.png')
	print(type(img),img.shape)
	img=torch.from_numpy(img)
	img=img.unsqueeze(dim=0)
	renet=video_feature_extraction('resnet18',512)
	output=renet(img)
	with open("output","w") as f:
		f.write(output)
	imgplot = plt.imshow(img)
	plt.show()
