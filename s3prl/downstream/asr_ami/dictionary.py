"""
dictionary.py — Token dictionary for CTC-based ASR.

Extends fairseq's Dictionary with multiprocessing-capable transcript counting,
used when building a character-level vocabulary from a corpus of transcripts.
"""

import os
from collections import Counter
from multiprocessing import get_context

import torch

from fairseq.data.dictionary import Dictionary as fairseq_Dictionary


class Dictionary(fairseq_Dictionary):
    """Character-level token dictionary backed by fairseq's Dictionary.

    Inherits all standard fairseq Dictionary methods (load, save, encode_line,
    bos, eos, pad, unk, index, string, etc.) and adds parallel transcript
    counting for efficient vocabulary construction.
    """

    @staticmethod
    def _add_transcripts_to_dictionary_single_worker(
        transcripts, eos_word, worker_id=0, num_workers=1
    ):
        """Count token frequencies in a shard of transcripts.

        Processes the slice of transcripts assigned to this worker and counts
        each whitespace-separated token, plus one occurrence of eos_word per line.

        Args:
            transcripts: Full list of transcript strings (all workers receive this).
            eos_word:    The EOS token string to append after each line.
            worker_id:   0-indexed ID of this worker (determines the shard).
            num_workers: Total number of workers (determines shard size).

        Returns:
            Counter mapping token → frequency for this worker's shard.
        """
        counter = Counter()
        size = len(transcripts)
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = min(size + 1, offset + chunk_size)
        for line in transcripts[offset:end]:
            for word in line.split():
                counter.update([word])
            counter.update([eos_word])
        return counter

    @staticmethod
    def add_transcripts_to_dictionary(transcripts, dict, num_workers):
        """Count tokens across all transcripts and add them to a Dictionary.

        Distributes the work across num_workers processes (using 'spawn' context
        to avoid fork-related issues), then merges the per-worker Counters into
        the dictionary via add_symbol.

        Args:
            transcripts: List of transcript strings to count tokens from.
            dict:        Dictionary instance to populate (modified in-place).
            num_workers: Number of parallel worker processes.
        """
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = get_context('spawn').Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_transcripts_to_dictionary_single_worker,
                        (transcripts, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_transcripts_to_dictionary_single_worker(
                    transcripts, dict.eos_word
                )
            )
