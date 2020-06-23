import argparse
import tensorflow as tf


from data_loader import load_data
from models import BaseCNNModel, AdvancedModel


parser = argparse.ArgumentParser(description='Captcha Model Training and Testing')
parser.add_argument('--data-path', '-i', type=str, default='../data/',
        help='path to input data')
parser.add_argument('--epochs', '-n', type=int, default=32,
        help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32,
        help='size of batch size')
parser.add_argument('--learning-rate', '--lr', type=float, default=1.,
        help='learning rate in training')
parser.add_argument('--model', default='base',
        help='Which model to run [base|advanced]')
parser.add_argument('--eval', type=bool, action='store_true',
        help='skip training (needed to specify model')
parser.add_argument('--load', type=str, default='',
        help='load model from an external file')
parser.add_argument('--save', type=str, default='auto',
        help='save model to an external file')
args = parser.parse_args()


def train(model, inputs):
    pass


def validate(model, inputs):
    pass


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if len(gpus) == 0:
        print('Warning: No GPUs present on this node, using CPU')
    else:
        # GPU memory is limited to 8 GBs here
        # Remove the following line if you want more GPU memory
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

    x_train, y_train, x_test, y_test = load_data(args.data_path, 0.9)
    model = AdvancedModel() if args.model == 'advanced' else BaseCNNModel()

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.fit(x_train, y_train, 
            epochs=args.epochs, 
            batch_size=args.batch_size)
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == '__main__':
    main()
