import os
import time
import argparse
import glob
import cv2
import numpy as np
import tensorflow as tf
from six.moves import xrange
from loader import Dataloader
from model import HairModel

parser = argparse.ArgumentParser(description='HairNet')

parser.add_argument('--mode',          type=str,   default='train')
parser.add_argument('--data_dir',      type=str,   default='../data')
parser.add_argument('--epochs',        type=int,   default=60)
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--output_dir',    type=str,   default='../model')
parser.add_argument('--result_dir',    type=str,   default='../result')
parser.add_argument('--upscale_level', type=int,   default=0)
parser.add_argument('--load_model',    action='store_true')

args = parser.parse_args()

def create_model(session):
    """
    Create model and initialize it or load its parameters in a session

    Args:
        session: tensorflow session
    Return:
        model: HairNet model (created / loaded)
    """
    model = HairModel(args.learning_rate, args.epochs, os.path.join(args.output_dir, 'summary'))

    if args.mode == 'train' and not args.load_model:
        print("Creating model with fresh parameters")
        session.run(tf.global_variables_initializer())
        return model

    # load a previously saved model
    ckpt_path = os.path.join(args.output_dir, 'ckpt')
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    if ckpt:
        print('loading model {}'.format(os.path.basename(ckpt)))
        model.saver.restore(session, ckpt)
        return model
    else:
        raise(ValueError, 'can NOT find model')

def evaluate_pos(pos, pos_gt, weight):
    """
    compute the Euclidean distance error for one hairstyle

    Args:
        pos: [32, 32, 300] output position
        pos_gt: [32, 32, 300] position ground truth
        weight: [32, 32, 100] whether the point is visible
    Return:
        average position error
    """
    square_loss = np.square(pos - pos_gt)
    error = np.zeros((32, 32, 100))  # Euclidean distance error
    for i in range(100):
        error[..., i] = np.sqrt(np.sum(square_loss[..., 3*i:3*i+3], axis=2))
    # compute visible error
    visible = weight
    visible_err = visible * error
    total_err = np.sum(visible_err) / np.sum(visible)

    return total_err


def train():
    """ just training """

    # load data
    print("loading test data")
    dataloader = Dataloader(args.data_dir, args.batch_size)
    train_samples = dataloader.train_samples
    test_samples = dataloader.test_samples

    train_batches = train_samples // args.batch_size
    test_batches = test_samples // args.batch_size
    
    print('total number of samples: {}'.format(train_samples))
    print('total number of steps: {}'.format(train_batches * args.epochs))

    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = create_model(sess)

        log_every_n_batches = 20
        start_time = time.time()

        for e in xrange(args.epochs):
            print('working on epoch {0}/{1}'.format(e + 1, args.epochs))
            epoch_start_time = time.time()
            epoch_loss, batch_loss = 0, 0
            dataloader.fresh_batch_order()

            for i in xrange(train_batches):
                if (i+1) % log_every_n_batches == 0:
                    print('working on epoch {0}, batch {1}/{2}'.format(e+1, i+1, train_batches))
                enc_in, dec_out = dataloader.get_train_batch(i)
                _, _, step_loss, summary = model.step(sess, enc_in, dec_out, True)
                epoch_loss += step_loss
                batch_loss += step_loss
                print(step_loss)
                model.train_writer.add_summary(summary, model.global_step.eval())
                if (i+1) % log_every_n_batches == 0:
                    print('current batch loss: {:.8f}'.format(batch_loss / log_every_n_batches))
                    batch_loss = 0

            epoch_time = time.time() - epoch_start_time
            print('epoch {0}/{1} finish in {2:.2f} s'.format(e+1, args.epochs, epoch_time))
            print('average epoch loss: {:.8f}'.format(epoch_loss / train_batches))

            print('saving model...')
            ckpt_path = os.path.join(args.output_dir, 'ckpt')
            model.saver.save(sess, os.path.join(ckpt_path, 'ckpt'), model.global_step.eval())

            # test after each epoch
            loss, pos_err = 0, 0
            for j in range(test_batches):
                test_x = []
                test_y = []
                visible_weight, test_x, test_y = dataloader.get_test_data(j)
                n = len(test_x)
                for k in range(n):
                    enc_in, dec_out = np.expand_dims(test_x[k], 0), np.expand_dims(test_y[k], 0)  # input must be [?, 32, 32, 500]
                    pos, curv, step_loss = model.step(sess, enc_in, dec_out, False)
                    step_pos_err = evaluate_pos(pos[0], test_y[k, ..., 100:400], visible_weight)
                    loss += step_loss
                    pos_err += step_pos_err
            avg_pos_err = pos_err / test_samples
            err_summary = sess.run(model.err_m_summary, {model.err_m: avg_pos_err})
            model.test_writer.add_summary(err_summary, model.global_step.eval())

            print('=================================\n'
                  'total loss avg:            %.8f\n'
                  'position error avg(m):     %.8f\n'
                  '=================================' % (loss / test_samples, avg_pos_err))

        print('training finish in {:.2f} s'.format(time.time() - start_time))

def write_strands(strands, upscale_level, file_name):
    if (3 != strands.ndim):
        return

    strands_1 = strands.reshape(strands.shape[0], strands.shape[1], strands.shape[2] // 3, 3)
    strands_2 = strands_1.transpose(0, 1, 3, 2)

    strands_x = strands_2[:, :, 0, :]
    strands_y = strands_2[:, :, 1, :]
    strands_z = strands_2[:, :, 2, :]

    for n in range(upscale_level):
        strands_x = cv2.resize(strands_x, (strands_x.shape[0] * 2, strands_x.shape[1] * 2), interpolation=cv2.INTER_LINEAR)
        strands_y = cv2.resize(strands_y, (strands_y.shape[0] * 2, strands_y.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
        strands_z = cv2.resize(strands_z, (strands_z.shape[0] * 2, strands_z.shape[1] * 2), interpolation=cv2.INTER_LINEAR)

    with open(file_name, 'wb') as file:
        strands_num = strands_x.shape[0] * strands_x.shape[1]
        file.write(np.uint32(strands_num))
        for i in range(strands_x.shape[0]):
            for j in range(strands_x.shape[1]):
                vertices_num = strands_x.shape[2]
                file.write(np.uint32(vertices_num))
                for k in range(vertices_num):
                    file.write(np.float32(strands_x[i][j][k]))
                    file.write(np.float32(strands_y[i][j][k]))
                    file.write(np.float32(strands_z[i][j][k]))

def gen_hair():
    """Generate hair strands"""

    print("loading data")
    input_dir = os.path.join(args.result_dir, 'input')
    input_files = glob.glob(os.path.join(input_dir, "*.exr"))
    output_dir = os.path.join(args.result_dir, 'output')
    output_files = []
    input_data = []
    for file in input_files:
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        input_data.append(image[:, :, (0, 1)])
        output_files.append(os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".data"))

    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = create_model(sess)

        print('generating hairs')
        for i in range(len(input_data)):
            data_x = np.expand_dims(input_data[i], 0)
            data_y = np.zeros((1, 32, 32, 500))
            pos, curv, _ = model.step(sess, data_x, data_y, False)
            write_strands(pos[0], args.upscale_level, output_files[i])

def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'gen_hair':
        gen_hair()

if __name__ == '__main__':
    main()
