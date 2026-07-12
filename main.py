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
    import io
    import numpy as np

    # google-cloud-vision reads GOOGLE_APPLICATION_CREDENTIALS itself; the client
    # picks it up from the environment, no hardcoded path needed here.
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
            x, y = float(vertices[:, 0].mean()), float(vertices[:, 1].mean())
            print(text.description, x, y)
            return x, y

    raise ValueError(f"keyword {keyword!r} not found in {path}")
