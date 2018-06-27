import torchvision, torch
import torchvision.transforms as transforms
from torchvision import transforms, datasets
#from nn_model_customFilterSize import net
#from nn_model import net
from learning import train_model
from torch.autograd import Variable # for computational graphs


# Parameter initialization
num_iteration = 200
learn_rate = 0.0001
mom = 0.9
	
transform = transforms.Compose([
	transforms.Grayscale(num_output_channels=1),
	transforms.Resize(size=(32,32), interpolation=2),
	transforms.ToTensor()
	#,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_im = 'data/train/'
train_img_database = datasets.ImageFolder(root=train_im, transform=transform)
# batch_size, num_workers = 1 always and shuffle=False. Bca we are loading images individually from the folder
train_dataset_loader = torch.utils.data.DataLoader(train_img_database, batch_size=1, 
													shuffle=False, num_workers=1)	

test_im = 'data/test/'
test_img_database = datasets.ImageFolder(root=test_im, transform=transform)
# batch_size, num_workers = 1 always and shuffle=False. Bcz we are loading images individually from the folder
test_dataset_loader = torch.utils.data.DataLoader(test_img_database, batch_size=1, 
													shuffle=False,num_workers=1)	
# Assign classes
classes = (0,1,2,3,4,5,6,7,8,9)



if __name__ == '__main__':

	print(net)
	train_model(train_dataset_loader, num_iteration, learn_rate, mom)
												
	correct = 0
	total = 0
	for j, data in enumerate(test_dataset_loader, 0):
		test_x = (data)[0]   # test image
		test_y = (data)[1]
		outputs = net(test_x)
		_, predicted = torch.max(outputs.data, 1)
		total += test_y.size(0)
		correct += (predicted == test_y).sum().item()

	print('Accuracy of the network on the 10 * 80 test images: %d %%' % (
		100 * correct / total))


	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))

	for j, data in enumerate(test_dataset_loader, 0):
		test_x = (data)[0]   # test image
		test_y = (data)[1]
		outputs = net(test_x)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == test_y).squeeze()
	#     for i in range(10):
		label = test_y.item()
		class_correct[label] += c.item()
		class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
	# print('Accuracy of the network on the 10 * 80 test images: %d %%' % (100 * correct / total))