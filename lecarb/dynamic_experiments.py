from ast import literal_eval
from time import time
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

from workload.gen_workload import generate_workload
from workload.gen_label import generate_labels, update_labels
from workload.merge_workload import merge_workload
from workload.dump_quicksel import dump_quicksel_query_files, generate_quicksel_permanent_assertions
from dataset.dataset import load_table, dump_table_to_num
from dataset.gen_dataset import generate_dataset
from dataset.manipulate_dataset import gen_appended_dataset
from estimator.sample import test_sample
# from estimator.postgres import test_postgres
# from estimator.mysql import test_mysql
from estimator.mhist import test_mhist
# from estimator.bayesnet import test_bayesnet
# from estimator.feedback_kde import test_kde
from estimator.utils import report_errors, report_dynamic_errors
from estimator.naru.naru import train_naru, test_naru, update_naru
from estimator.mscn.mscn import train_mscn, test_mscn
# from estimator.lw.lw_nn import train_lw_nn, test_lw_nn
# from estimator.lw.lw_tree import train_lw_tree, test_lw_tree
from estimator.deepdb.deepdb import train_deepdb, test_deepdb, update_deepdb
from workload.workload import dump_sqls


def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', default=2)
    args.add_argument('--seed', default=123)

    args.add_argument('--pattern', default='update-train', choices=['workload', 'dataset', 'train', 'test', 'report',
                                                             'report-dynamic', 'update-train'])
    args.add_argument('--w_pattern', default='gen', choices=['gen', 'label', 'update-label', 'merge', 'quicksel',
                                                             'dump', 'convert'])
    args.add_argument('--d_pattern', default='table', choices=['table', 'gen', 'update', 'dump'])
    args.add_argument('--model_name')
    args.add_argument('--workload', default='dynamic')
    args.add_argument('--dataset', default='dypower5')
    args.add_argument('--dataset_version', default='update')
    args.add_argument('--estimator', default='naru', choices=['deepdb', 'naru', 'mhist', 'sample'])
    args.add_argument('--params')
    args.add_argument('--sizelimit', default=0)
    args.add_argument('--overwrite', default=True)
    args.add_argument('--old_version', default='')
    args.add_argument('--win_ratio', default='')
    args.add_argument('--no_label', default=False)
    args.add_argument('--g_pattern', default='base', choices=['base', 'train', 'test'])
    args.add_argument('--file_name', default='base-resmade_hid32,32,32,32_emb4_ep20_embedInembedOut_warm0-123')
    args.add_argument('--ap_size', default=0.2)
    args.add_argument('--ap_type', default='ind', choices=['ind', 'skew', 'cor'])
    args.add_argument('--sample_ratio', default=1.0)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # set_gpu(gpus=args.gpu)

    seed = args.seed
    args.ap_size = float(args.ap_size)
    if seed is None:
        seed = int(time())
    else:
        seed = int(seed)

    if args.pattern == 'workload':
        if args.w_pattern == 'gen':
            if args.g_pattern == 'base':
                n_train = 10
                n_valid = 2
                n_test = 10
            elif args.g_pattern == 'train':
                n_train = 10
                n_valid = 2
                n_test = 0
            elif args.g_pattern == 'test':
                n_train = 0
                n_valid = 0
                n_test = 1000
            else:
                raise NotImplementedError
            params = {
                'attr': {'pred_number': 1.0},
                'center': {'distribution': 0.9, 'vocab_ood': 0.1},
                'width': {'uniform': 0.5, 'exponential': 0.5},
                'number': {'train': n_train, 'valid': n_valid, 'test': n_test}
            }
            generate_workload(
                seed,
                dataset=args.dataset,
                version=args.dataset_version,
                name=args.workload,
                no_label=args.no_label,
                old_version=args.old_version,
                win_ratio=args.win_ratio,
                params=params
            )
        elif args.w_pattern == "label":
            generate_labels(
                dataset=args.dataset,
                version=args.dataset_version,
                workload=args.workload
            )
        elif args.w_pattern == "update-label":
            update_labels(
                seed,
                dataset=args.dataset,
                version=args.dataset_version,
                workload=args.workload,
                sampling_ratio=args.sample_ratio
            )
        elif args.w_pattern == "merge":
            merge_workload(
                dataset=args.dataset,
                version=args.dataset_version,
                workload=args.workload
            )
        elif args.w_pattern == "quicksel":
            dump_quicksel_query_files(
                dataset=args.dataset,
                version=args.dataset_version,
                workload=args.workload,
                overwrite=args.overwrite
            )
            generate_quicksel_permanent_assertions(
                dataset=args.dataset,
                version=args.dataset_version,
                params=literal_eval(args.params),
                overwrite=args.overwrite
            )
        elif args.w_pattern == "dump":
            dump_sqls(
                dataset=args.dataset,
                version=args.dataset_version,
                workload=args.workload)
        else:
            raise NotImplementedError
        exit(0)

    if args.pattern == 'dataset':
        if args.d_pattern == "table":
            load_table(args.dataset, args.dataset_version, overwrite=args.overwrite)
        elif args.d_pattern == "gen":
            generate_dataset(
                seed,
                dataset=args.dataset,
                version=args.dataset_version,
                params=literal_eval(args.params),
                overwrite=args.overwrite
            )
        elif args.d_pattern == "update":
            params = {
                'type': args.ap_type,
                'batch_ratio': args.ap_size
            }
            gen_appended_dataset(
                seed,
                dataset=args.dataset,
                version=args.dataset_version,
                params=params,
                overwrite=args.overwrite
            )
        elif args.d_pattern == "dump":
            dump_table_to_num(args.dataset, args.dataset_version)
        else:
            raise NotImplementedError
        exit(0)

    if args.pattern == "train":
        dataset = args.dataset
        version = args.dataset_version
        workload = args.workload
        # params = literal_eval(args.params)
        sizelimit = float(args.sizelimit)

        if args.estimator == "naru":
            params = {
                'epochs': 20,
                'input_encoding': 'embed',
                'output_encoding': 'embed',
                'embed_size': 4,
                'layers':  4,
                'fc_hiddens': 32,
                'residual': True,
                'warmups': 0
            }
            train_naru(seed, dataset, version, workload, params, sizelimit)
        elif args.estimator == "mscn":
            params = {
                'epochs': 200,
                'bs': 1024,
                'num_samples': 1000,
                'hid_units': 16,
                'train_num': 100000
            }
            train_mscn(seed, dataset, version, workload, params, sizelimit)
        elif args.estimator == "deepdb":
            params = {
                'hdf_sample_size': 100000,
                'rdc_threshold': 0.3,
                'ratio_min_instance_slice': 0.01
            }
            train_deepdb(seed, dataset, version, workload, params, sizelimit)
        elif args.estimator == "lw_nn":
            params = {
                'epochs': 100,
                'bins': 200,
                'hid_units': '128_64_32',
                'train_num': 10000,
                'bs': 32
            }
            train_lw_nn(seed, dataset, version, workload, params, sizelimit)
        elif args.estimator == "lw_tree":
            params = {
                'trees': 16,
                'bins': 200,
                'train_num': 10000
            }
            train_lw_tree(seed, dataset, version, workload, params, sizelimit)
        else:
            raise NotImplementedError
        exit(0)

    if args.pattern == "test":
        dataset = args.dataset
        version = args.dataset_version
        workload = args.workload
        # params = literal_eval(args.params)
        overwrite = args.overwrite

        if args.estimator == "sample":
            params = {
                'version': 'original',
                'ratio': 0.015
            }
            test_sample(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "postgres":
            params = {
                'version': 'original',
                'stat_target': 10000
            }
            test_postgres(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "mysql":
            params = {
                'version': 'original',
                'bucket': 1024
            }
            test_mysql(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "mhist":
            params = {
                'version': 'original',
                'num_bins': 30000
            }
            test_mhist(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "bayesnet":
            params = {
                'version': 'original',
                'samples': 200,
                'discretize': 100,
                'parallelism': 50
            }
            test_bayesnet(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "kde":
            params = {
                'version': 'original',
                'ratio': 0.015,
                'train_num': 10000
            }
            test_kde(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "naru":
            params = {
                'psample': 2000,
                'model': args.model_name
            }
            test_naru(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "mscn":
            params = {
                'model': args.model_name
            }
            test_mscn(dataset, version, workload, params, overwrite)
        elif args.estimator == "deepdb":
            params = {
                'model': args.model_name
            }
            test_deepdb(dataset, version, workload, params, overwrite)
        elif args.estimator == "lw_nn":
            params = {
                'model': args.model_name,
                'use_cache': True
            }
            test_lw_nn(dataset, version, workload, params, overwrite)
        elif args.estimator == "lw_tree":
            params = {
                'model': args.model_name,
                'use_cache': True
            }
            test_lw_tree(dataset, version, workload, params, overwrite)
        else:
            raise NotImplementedError
        exit(0)

    if args.pattern == "report":
        dataset = args.dataset
        params = {
            'file': args.file_name
        }
        report_errors(dataset, params['file'])
        exit(0)

    if args.pattern == "report-dynamic":
        dataset = args.dataset
        params = literal_eval(args.params)
        report_dynamic_errors(dataset, params['old_new_file'], params['new_new_file'], params['T'],
                              params['update_time'])
        exit(0)

    if args.pattern == "update-train":
        dataset = args.dataset
        version = args.dataset_version
        workload = args.workload
        # params = literal_eval(args.params)
        overwrite = args.overwrite
        # overwrite is not work in such case

        if args.estimator == "naru":
            params = {
                'model': args.model_name,
                'epochs': 1
            }
            update_naru(seed, dataset, version, workload, params, overwrite)
        elif args.estimator == "deepdb":
            params = {
                'model': args.model_name
            }
            update_deepdb(seed, dataset, version, workload, params, overwrite)
        else:
            raise NotImplementedError
        exit(0)
