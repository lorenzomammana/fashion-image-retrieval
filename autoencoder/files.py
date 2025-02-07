from pathlib import Path

ROOT=Path('/home/ubuntu/fashion-dataset')
autoencoder_path=ROOT / 'autoencoder.h5'
encoder_path=ROOT / 'encoder.h5'
decoder_path=ROOT / 'decoder.h5'
small_images_directory = ROOT / 'small'
small_images_classes_directory = ROOT / 'small_classes'
output_directory = ROOT / 'output'
classes_distribution_pdf_path = output_directory / 'classes_distribution.pdf'
