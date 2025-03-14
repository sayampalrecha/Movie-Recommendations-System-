import streamlit as st
import pickle
import pandas as pd
from scipy import sparse
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #FF4B4B;
    }
    .recommendation-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stProgress > div > div > div {
        background-color: #FF4B4B;
    }
    .sidebar-content {
        padding: 20px;
    }
    h1 {
        color: #FF4B4B;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title('🎬 Movie Recommender System')
st.markdown("### Discover movies similar to your favorites")


# Load movie data and similarity matrix
@st.cache_resource
def load_data():
    try:
        # Load movie data
        movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)

        # Try to load sparse similarity matrix first (if available)
        try:
            similarity = sparse.load_npz('similarity_sparse.npz')
            is_sparse = True
        except:
            # If sparse matrix not found, load regular similarity matrix
            similarity = pickle.load(open('similarity1.pkl', 'rb'))
            is_sparse = False

        return movies, similarity, is_sparse
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


movies, similarity, is_sparse = load_data()

if movies is not None and similarity is not None:
    # Function to recommend movies
    def recommend(movie):
        try:
            movie_index = movies[movies['title'] == movie].index[0]

            # Get similarity scores
            if is_sparse:
                # For sparse matrix
                distances = similarity[movie_index].toarray()[0]
            else:
                # For regular matrix
                distances = similarity[movie_index]

            # Get top 5 similar movies (excluding the movie itself)
            movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

            recommended_movies = []
            for i in movie_indices:
                recommended_movies.append({
                    'title': movies.iloc[i[0]].title,
                    'similarity_score': min(float(i[1]) * 100, 100)  # Cap at 100%
                })

            return recommended_movies
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            return []


    # Sidebar information
    with st.sidebar:
        st.markdown("## About This App")
        st.markdown("""
        This movie recommender system uses **content-based filtering** to suggest movies similar to your selection.

        ### How it works:
        1. Select a movie you enjoy
        2. Click "Get Recommendations"
        3. The system analyzes the movie's features:
           - Genre
           - Cast
           - Director
           - Keywords
           - Plot elements
        4. It identifies movies with similar characteristics

        ### Technology:
        The recommendations are based on cosine similarity between movie feature vectors, which measures how similar two movies are based on their content.
        """)

    # Main content area
    # Movie selector (alphabetically sorted)
    movie_options = sorted(movies['title'].values)
    selected_movie = st.selectbox('Select a movie you like:', movie_options)

    # Recommendation button
    if st.button('Get Recommendations', use_container_width=False):
        with st.spinner('Finding similar movies...'):
            recommendations = recommend(selected_movie)

            if recommendations:
                # Clear previous results by using a container
                st.markdown("## Top Recommendations")
                st.markdown(f"### Based on your selection: {selected_movie}")
                st.markdown("---")

                # Display each recommendation in its own row
                for i, movie in enumerate(recommendations):
                    st.markdown(f"### {i + 1}. {movie['title']}")
                    st.progress(movie['similarity_score'] / 100)
                    st.markdown(f"**Similarity Score:** {movie['similarity_score']:.1f}%")
                    st.markdown("---")
            else:
                st.warning("No recommendations found. Please try another movie.")

    # Show a message if no recommendations have been made yet
    if 'recommendations' not in locals():
        st.info("👆 Select a movie and click 'Get Recommendations' to discover similar films")

        # Show some popular movies as examples
        st.markdown("### Popular Selections")
        popular_movies = ["Avatar", "Titanic", "The Dark Knight", "Inception", "Pulp Fiction"]

        # Display popular movies in a horizontal layout with buttons
        cols = st.columns(len(popular_movies))
        for i, movie in enumerate(popular_movies):
            if movie in movie_options:
                with cols[i]:
                    if st.button(movie, key=f"popular_{i}"):
                        st.session_state.selected_movie = movie
                        # Rerun to update the selectbox
                        st.experimental_rerun()
else:
    st.error("Failed to load movie data. Please check if the required pickle files are in the correct location.")
    st.info("""
    ### Required files:
    - `movies_dict.pkl` - Movie data dictionary
    - One of the following:
        - `similarity.pkl` - Regular similarity matrix
        - `similarity_sparse.npz` - Compressed sparse similarity matrix
    """)
