import torch.optim as optim
import torch.nn as nn
from nn_model import net
from torch.autograd import Variable # for computational graphs

def train_model(train_data, tr_epoch, len_rate, nn_mom):
	train_dataset_loader = train_data
	
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=len_rate, momentum=nn_mom)

	for epoch in range(tr_epoch):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(train_dataset_loader, 0):
				#get the inputs
				inputs = (data)[0]   # image
				labels = (data)[1]   # Label

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				# print statistics
				running_loss += loss.item()
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0

	print('Finished Training')