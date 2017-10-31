import __init__

from FunctionUtils import sync_real_time_counter
import Main2Parser


@sync_real_time_counter('test')
def exec_query(data_path, parser):
    Main2Parser.exec_query(data_path, parser)
