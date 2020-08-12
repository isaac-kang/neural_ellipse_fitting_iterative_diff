import tensorflow as tf 

def get_flags(MAIN_DIR):

    tf.app.flags.DEFINE_integer('summary_every', 10, 'summary iteration period')
    tf.app.flags.DEFINE_integer('save_examples_every', 1000, 'Sets how often to compute target predictions and write them to the experiments directory.')
    tf.app.flags.DEFINE_integer('save_every', 1000, 'How many iterations to do per save.')
    tf.app.flags.DEFINE_integer('eval_every', 5000, 'How many iterations to do per calculating the loss on the test set. This operation is time-consuming, so should not be done often.')
    tf.app.flags.DEFINE_integer('save_imgs_every', 1000, 'How many iterations to do per img save.')

    
    tf.app.flags.DEFINE_integer('gpu', 0, 'gpu')
    tf.app.flags.DEFINE_string('mode', 'train', 'Options: {train,eval,predict}.')
    tf.app.flags.DEFINE_string('experiment_name',  'default_name', 'name of experiment')
    tf.app.flags.DEFINE_string('test_dir', '', '../data/test/?')
    tf.app.flags.DEFINE_integer('keep', 3, 'num saved checkpoints')
    tf.app.flags.DEFINE_integer('print_every', 500, 'print')
<<<<<<< HEAD
    tf.app.flags.DEFINE_integer('num_summary_images', 1, 'How many images to write to summary.')
=======
    tf.app.flags.DEFINE_integer('num_summary_images', 2, 'How many images to write to summary.')
>>>>>>> 629b1c860452b2d8d5a7383fbf954654de87ba9b


    # Path
    tf.app.flags.DEFINE_string('train_dir', '', 'Sets the dir to which checkpoints and logs will be saved. Defaults to output/{experiment_name}.')
    tf.app.flags.DEFINE_string('save_imgs_dir', './train_results', 'train results dir')
    tf.app.flags.DEFINE_list('data_dir', '../data/coco/val2017', 'dataset dir list')
    tf.app.flags.DEFINE_list('validation_dataset_file_path', '', 'validation tfrecords list')

    # Train Options
    tf.app.flags.DEFINE_integer('batch_size', 8,  'Minibatch size')
<<<<<<< HEAD
    tf.app.flags.DEFINE_integer('num_preprocessing_processes', 1, 'The number of processes to create batches.')
=======
    tf.app.flags.DEFINE_integer('num_preprocessing_processes', 4, 'The number of processes to create batches.')
>>>>>>> 629b1c860452b2d8d5a7383fbf954654de87ba9b
    tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate')
    tf.app.flags.DEFINE_integer('num_samples_per_learning_rate_half_decay', 1600000, 'Number of samples for half decaying learning rate')
    tf.app.flags.DEFINE_float('lr_decay_every', 30000, 'Sets the intervals at which to do learning rate decay (cuts by 1/2). Setting to 0 means no decay')
    tf.app.flags.DEFINE_boolean('data_augmentation', True, 'Sets whether or not to perform data augmentation')
<<<<<<< HEAD
    tf.app.flags.DEFINE_integer('num_iterative_structure', 4, 'number of iteration steps for the model')
=======
>>>>>>> 629b1c860452b2d8d5a7383fbf954654de87ba9b
    

    return tf.app.flags.FLAGS
