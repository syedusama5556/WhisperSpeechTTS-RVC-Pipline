"""create dataset and dataloader"""
import torch
import torch.utils.data
from munch import munchify

from dlas.utils.util import opt_get


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, collate_fn=None, shuffle=True):
    phase = dataset_opt['phase']
    pin_memory = opt_get(dataset_opt, ['pin_memory'], True)
    if phase == 'train':
        if opt_get(opt, ['dist'], False):
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
        else:
            num_workers = dataset_opt['n_workers']
            batch_size = dataset_opt['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=pin_memory, collate_fn=collate_fn, persistent_workers=True)
    else:
        batch_size = dataset_opt['batch_size'] or 1
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                           pin_memory=pin_memory, collate_fn=collate_fn)


def create_dataset(dataset_opt, return_collate=False):
    mode = dataset_opt['mode']
    collate = None

    # datasets for image restoration
    if mode == 'fullimage':
        from dlas.data.images.full_image_dataset import FullImageDataset as D
    elif mode == 'single_image_extensible':
        from dlas.data.images.single_image_dataset import \
            SingleImageDataset as D
    elif mode == 'multi_frame_extensible':
        from dlas.data.images.multi_frame_dataset import MultiFrameDataset as D
    elif mode == 'combined':
        from dlas.data.combined_dataset import CombinedDataset as D
    elif mode == 'multiscale':
        from dlas.data.images.multiscale_dataset import MultiScaleDataset as D
    elif mode == 'paired_frame':
        from dlas.data.images.paired_frame_dataset import \
            PairedFrameDataset as D
    elif mode == 'stylegan2':
        from dlas.data.images.stylegan2_dataset import Stylegan2Dataset as D
    elif mode == 'imagefolder':
        from dlas.data.images.image_folder_dataset import \
            ImageFolderDataset as D
    elif mode == 'torch_dataset':
        from data.torch_dataset import TorchDataset as D
    elif mode == 'byol_dataset':
        from dlas.data.images.byol_attachment import ByolDatasetWrapper as D
    elif mode == 'byol_structured_dataset':
        from dlas.data.images.byol_attachment import \
            StructuredCropDatasetWrapper as D
    elif mode == 'random_aug_wrapper':
        from dlas.data.images.byol_attachment import \
            DatasetRandomAugWrapper as D
    elif mode == 'random_dataset':
        from dlas.data.images.random_dataset import RandomDataset as D
    elif mode == 'zipfile':
        from dlas.data.images.zip_file_dataset import ZipFileDataset as D
    elif mode == 'nv_tacotron':
        from dlas.data.audio.nv_tacotron_dataset import TextMelCollate as C
        from dlas.data.audio.nv_tacotron_dataset import TextWavLoader as D
        from dlas.models.audio.tts.tacotron2 import create_hparams
        default_params = create_hparams()
        default_params.update(dataset_opt)
        dataset_opt = munchify(default_params)
        if opt_get(dataset_opt, ['needs_collate'], True):
            collate = C()
    elif mode == 'paired_voice_audio':
        from dlas.data.audio.paired_voice_audio_dataset import \
            TextWavLoader as D
        from dlas.models.audio.tts.tacotron2 import create_hparams
        default_params = create_hparams()
        default_params.update(dataset_opt)
        dataset_opt = munchify(default_params)
    elif mode == 'fast_paired_voice_audio':
        from dlas.data.audio.fast_paired_dataset import \
            FastPairedVoiceDataset as D
        from dlas.models.audio.tts.tacotron2 import create_hparams
        default_params = create_hparams()
        default_params.update(dataset_opt)
        dataset_opt = munchify(default_params)
    elif mode == 'fast_paired_voice_audio_with_phonemes':
        from dlas.data.audio.fast_paired_dataset_with_phonemes import \
            FastPairedVoiceDataset as D
        from dlas.models.audio.tts.tacotron2 import create_hparams
        default_params = create_hparams()
        default_params.update(dataset_opt)
        dataset_opt = munchify(default_params)
    elif mode == 'gpt_tts':
        from dlas.data.audio.gpt_tts_dataset import GptTtsCollater as C
        from dlas.data.audio.gpt_tts_dataset import GptTtsDataset as D
        collate = C(dataset_opt)
    elif mode == 'unsupervised_audio':
        from dlas.data.audio.unsupervised_audio_dataset import \
            UnsupervisedAudioDataset as D
    elif mode == 'unsupervised_audio_with_noise':
        from dlas.data.audio.audio_with_noise_dataset import \
            AudioWithNoiseDataset as D
    elif mode == 'preprocessed_mel':
        from dlas.data.audio.preprocessed_mel_dataset import \
            PreprocessedMelDataset as D
    elif mode == 'grand_conjoined_voice':
        from dlas.data.audio.grand_conjoined_dataset import \
            GrandConjoinedDataset as D
        from dlas.data.zero_pad_dict_collate import ZeroPadDictCollate as C
        if opt_get(dataset_opt, ['needs_collate'], False):
            collate = C()
    else:
        raise NotImplementedError(
            'Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    if return_collate:
        return dataset, collate
    else:
        return dataset


def get_dataset_debugger(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'paired_voice_audio':
        from dlas.data.audio.paired_voice_audio_dataset import \
            PairedVoiceDebugger
        return PairedVoiceDebugger()
    elif mode == 'fast_paired_voice_audio':
        from dlas.data.audio.fast_paired_dataset import FastPairedVoiceDebugger
        return FastPairedVoiceDebugger()
    return None
