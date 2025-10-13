from django.urls import path
from .views import search_recommendations

urlpatterns = [
    # path('recommend/<int:user_id>/', get_recommendations, name='recommend'),
    path('', search_recommendations, name='search_recommendations'),
]
