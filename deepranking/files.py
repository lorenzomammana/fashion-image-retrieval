from pathlib import Path

ROOT=Path('/home/ubuntu/fashion-dataset')
output_directory = ROOT / 'output'
autoencoder_path=ROOT / 'autoencoder.h5'
encoder_path=ROOT / 'encoder.h5'
decoder_path=ROOT / 'decoder.h5'
small_images_directory = ROOT / 'small'
small_images_classes_directory = ROOT / 'small_classes'
deepranking_weights_path = output_directory / 'deepranking.h5'
classes_distribution_pdf_path = output_directory / 'classes_distribution.pdf'
