import requests
import pprint

# cat 
def get_cat_image():
    url = ("https://api.thecatapi.com/v1/images/search"  
            + "?size=med&mime_types=jpg&format=json&has_breeds=true"
            + "&order=RANDOM&page=0&limit=1")
    response = requests.get(url)
    response_data = response.json()
    if response.status_code == 200:
        image_url = response_data[0]["url"]
        response = requests.get(image_url)
        return response.content     

def save_cat_image():
    with open("cat.jpg", "wb") as f:
        img = get_cat_image()
        f.write(img)

# dog 
def get_dog_image():
    url = ("https://api.thedogapi.com/v1/images/search"  
            + "?size=med&mime_types=jpg&format=json&has_breeds=true"
            + "&order=RANDOM&page=0&limit=1")
    response = requests.get(url)
    response_data = response.json()
    if response.status_code == 200:
        image_url = response_data[0]["url"]
        response = requests.get(image_url)
        return response.content

# TODO: Use a local cache for downloaded unsplash images (e.g. 5 images)

# unsplash
def get_unsplash_image(query_str):
    url = ("https://api.unsplash.com/photos/random"
           + f"?query={query_str}&count=1"
           + "&client_id=YOUR_API_KEY")
    response = requests.get(url)
    response_data = response.json()
    print("human response")
    pprint.pprint(response_data)
    if response.status_code == 200:
        image_url = response_data[0]["urls"]["regular"]
        response = requests.get(image_url)
        return response.content
    pass