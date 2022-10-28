def detect_text(path, keyword=''):
    """
    Detects text coordinates in the Image.
    
    Parameters:
      path: Image Path
      keyword: word in image to get coordinates of it.
      
    returns:
      X, Y coordinates of text(keyword) in that image.
    """
    
    from google.cloud import vision
    import io, os
    import numpy as np

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'cp-vision-905c8b772ee7.json'
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        if text.description.strip() == keyword:
            vertices = np.array([[vertex.x, vertex.y]
                        for vertex in text.bounding_poly.vertices])
            print(text.description, sum(vertices[:,:1])/4, sum(vertices[:,1:])/4)
    return sum(vertices[:,:1])/4, sum(vertices[:,1:])/4
