from collections import namedtuple
import unittest

from src.dlio_postprocessor import DLIOPostProcessor

class TestDLIOPostProcessor(unittest.TestCase):

    def create_DLIO_PostProcessor(self, args):
        return DLIOPostProcessor(args)

    def test_process_loading_and_processing_times(self):
        args = {
            'output_folder': 'tests/test_data',
        }
        args = namedtuple('args', args.keys())(*args.values())
        postproc = self.create_DLIO_PostProcessor(args)

        postproc.process_loading_and_processing_times()

        # Expected values: {
        #   'samples/s': {'mean': '3.27', 'std': '2.39', 'min': '1.33', 'median': '2.33', 'p90': '7.60', 'p99': '8.00', 'max': '8.00'}, 
        #   'sample_latency': {'mean': '3.27', 'std': '2.39', 'min': '1.33', 'median': '2.33', 'p90': '7.60', 'p99': '8.00', 'max': '8.00'}, 
        #   'avg_process_loading_time': '21.00', 
        #   'avg_process_processing_time': '21.00'
        # }
        self.assertEqual(postproc.overall_stats['samples/s']['mean'], '3.27')
        self.assertEqual(postproc.overall_stats['sample_latency']['mean'], '3.27')
        self.assertEqual(postproc.overall_stats['avg_process_loading_time'], '21.00')
        self.assertEqual(postproc.overall_stats['avg_process_processing_time'], '21.00')



if __name__ == '__main__':
    unittest.main()
