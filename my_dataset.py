from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class MNIST:

    def __init__(self, batch_size=100):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root="../MNIST_data", train=True, transform=transform)
        test_dataset = datasets.MNIST(root="../MNIST_data", train=False, transform=transform)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


if __name__ == '__main__':

    mnist = MNIST()

    for index, (xs, ys) in enumerate(mnist.test_dataloader):
        print(index)
