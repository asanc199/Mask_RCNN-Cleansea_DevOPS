import argparse



PRETRAIN_OPTIONS = ['coco', 'imagenet', 'last']
AUGMENTATION_OPTIONS = ['none', 'mild', 'severe']


"""Arguments menu"""
def menu():
	parser = argparse.ArgumentParser(description='Sequence-to-sequence Contrastive Learning')

	parser.add_argument('-pretrain',    dest="pretrain",                required = False,			help='Pretrain corpus', choices = PRETRAIN_OPTIONS, default = PRETRAIN_OPTIONS[0])
	# parser.add_argument('-epochs',		dest="epochs",                  required = True,			help='Epochs', type = int)
	parser.add_argument('-aug',         dest="augmentation",            required = True,			help='Augmentation type', choices = AUGMENTATION_OPTIONS)
	parser.add_argument('-size',        dest="size_perc",               required = False,			help='Train size percentage', default = 100, type = int)
	parser.add_argument('-epochs',		dest='epochs',					required = True,			help='List for the epoch breaks', type=str, default = '50, 100')


	args = parser.parse_args()
	args.epochs = [int(item) for item in args.epochs.split(',')]

	return args

