import time
import os
from training_options import TrainOptions
from dataset_loader import CreateDataLoader
from get_modelname import create_model
from visualizer import Visualizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

if __name__=='__main__':
    options = TrainOptions().parse()
    dataloader = CreateDataLoader(options)
    dataset = dataloader.load_data()
    dataset_size = dataloader.__len__()
    print('*** Training Image Count = %d ***' % dataset_size)

    #weightage_gp = 10.0

    model = create_model(options)
    model.setup(options)
    visualizer = Visualizer(options)

    total_steps = 0
    #regularizer_weight = options.regularizer_weight

    for epoch in range(options.epoch_count, (options.niter + options.niter_decay + 1)):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for iter_count, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % options.print_freq == 0:
                t_data = (iter_start_time - iter_data_time)
            visualizer.reset()
            total_steps += options.batchSize
            epoch_iter += options.batchSize
            model.set_input(data)
            #model.optimize_parameters(options, weightage_gp)
            #if total_steps % options.regularizer_decay == 0:
            #    regularizer_weight = (regularizer_weight * 0.5)

            model.optimize_parameters()

            if total_steps % options.display_freq == 0:
                save_result = total_steps % options.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % options.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / options.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if options.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, options, losses)

            if total_steps % options.save_latest_freq == 0:
                print('Saving the latest model [Epoch %d, Total Steps %d]' % (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % options.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, (options.niter + options.niter_decay), (time.time() - epoch_start_time)))
        model.update_learning_rate()











