import argparse
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf


from data_loader import load_data
from models import BaseCNNModel, AdvancedModel


parser = argparse.ArgumentParser(description='Captcha Model Training and Testing')
parser.add_argument('--data-path', '-i', type=str, default='../data/',
        help='path to input data')
parser.add_argument('--epochs', '-n', type=int, default=16,
        help='number of epochs to train')
parser.add_argument('--model', default='base',
        help='Which model to run [base|advanced]')
parser.add_argument('--eval', action='store_true',
        help='skip training (needed to specify model')
parser.add_argument('--load', type=str, default='',
        help='load model from an external file')
parser.add_argument('--save', type=str, default='auto',
        help='save model to an external file')
args = parser.parse_args()


def accuracy(test_labels, predict_labels):
    y = tf.cast(tf.equal(tf.argmax(test_labels, -1), 
        tf.argmax(predict_labels, -1)), tf.float32)
    acc = tf.reduce_mean(tf.reduce_prod(y, -1))
    return acc


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if len(gpus) == 0:
        print('Warning: No GPUs present on this node, using CPU')
    else:
        # GPU memory is limited to 3 GBs here
        # Remove the following line if you want more GPU memory
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])

    x_train, y_train, x_test, y_test = load_data(args.data_path, 0.9)
    model = AdvancedModel() if args.model == 'advanced' else BaseCNNModel()

    # TODO for task #2: You may modify training setup here
    model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[accuracy])

    if args.load != '':
        model.load_weights(args.load)

    if not args.eval:
        model.fit(x_train, y_train, epochs=args.epochs)
    loss, acc = model.evaluate(x_test,  y_test, verbose=2)

    if args.save == 'auto':
        args.save = 'ckpt/model-{}-at-{}'.format(args.model, 
                time.strftime('%y%m%d-%H%M%S'))

    if args.save != 'none' and not args.eval:
        print('Your model is saved to {}'.format(args.save))
        model.save_weights(args.save)

    if args.save != 'none':
        with open('{}.report'.format(args.save), 'w') as f:
            f.write('{}\n'.format(' '.join(sys.argv)))
            f.write('loss={}\nacc={}'.format(loss, acc))


if __name__ == '__main__':
    main()
