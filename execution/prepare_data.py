if __name__ == '__main__':
    from utilities import make_path, check_existence
    from config import dataset
    from model.preparation.pre_processing import process_documents

    content_index = -1

    base_path = make_path('data/source/')
    source_path = base_path / (dataset + '.csv')
    cleaned_path = base_path / (dataset + '_clean.csv')

    check_existence(source_path)
    print('Config complete.')

    process_documents(source_path, cleaned_path, content_index=content_index)
