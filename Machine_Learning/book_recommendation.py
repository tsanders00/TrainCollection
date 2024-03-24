from sklearn.neighbors import NearestNeighbors
import pandas as pd

books = pd.read_csv('./book-crossings/BX-Books.csv', sep=';',
                    encoding='iso-8859-1', on_bad_lines='skip', low_memory=False)
book_ratings = pd.read_csv('./book-crossings/BX-Book-Ratings.csv', sep=';',
                           encoding='iso-8859-1', on_bad_lines='skip', low_memory=False)

books = books.dropna()
book_ratings = book_ratings.dropna()
books = books.drop(['Publisher', 'Book-Author', 'Year-Of-Publication', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'],
                   axis=1)
# basically add book ratings to book df
data = pd.merge(books, book_ratings, on='ISBN', how='left')

# filter
# remove books with less than 100 reviews
isbn_keep = data.groupby('ISBN').size()
isbn_keep = isbn_keep[isbn_keep.values > 100]
data = data[data['ISBN'].isin(isbn_keep.index)]

# prep data for algorithm
data = data.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
# replace ISBN with book title but keeps number in background
data.index = data.join(books.set_index('ISBN'))['Book-Title']
data = data.sort_index()

# init model and fit
neighbor = NearestNeighbors(n_neighbors=5)
neighbor.fit(data.values)

def get_recommendations(book_title: str, neighbor: NearestNeighbors, data: pd.DataFrame):
    """

    :param book_title:
    :param neighbor:
    :param data:
    :return: None
    """
    # use book ratings to find nearest books
    distances, idxs = neighbor.kneighbors(X=[data.loc[book_title].values], n_neighbors=6, return_distance=True)
    print(f' The nearest books are: {data.iloc[idxs[0][1:]].index.values}')

get_recommendations('The Queen of the Damned (Vampire Chronicles (Paperback))', neighbor=neighbor, data=data)