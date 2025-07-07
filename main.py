import torch
import logging
import argparse
import json
import eval
import os
from prompt_generator import generate_dictionary
from image_generator import generate_images
from datasets.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from geoclip import GeoCLIP

logging.basicConfig(
    filename='geoclip.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

city_dict = {'New York City skyline at sunset, showing Manhattan skyscrapers and Hudson River': (40.7128, -74.006), 'Historic London street with red telephone booths and Big Ben in the background': (51.5074, -0.1278), 'Eiffel Tower and Parisian cafes in autumn, romantic atmosphere': (48.8566, 2.3522), 'Shibuya Crossing in Tokyo at night, neon lights and crowds': (35.6895, 139.6917), 'Ancient Roman Forum ruins and Colosseum in Rome under a dramatic sky': (41.9028, 12.4964), 'Brandenburg Gate and Reichstag building in Berlin, modern urban scene': (52.52, 13.405), 'Sydney Opera House and Harbour Bridge with fireworks, festive scene': (-33.8688, 151.2093), 'Christ the Redeemer statue overlooking Rio de Janeiro, panoramic view': (-22.9068, -43.1729), 'Pyramids of Giza and Sphinx near Cairo, desert landscape': (30.0444, 31.2357), 'Forbidden City and Tiananmen Square in Beijing, majestic architecture': (39.9042, 116.4074), "St. Basil's Cathedral in Red Square, Moscow, vibrant colors": (55.7558, 37.6173), 'Burj Khalifa and modern architecture in Dubai at twilight': (25.2769, 55.2963), 'Gardens by the Bay and Marina Bay Sands in Singapore, futuristic look': (1.3521, 103.8198), 'Canals and historic gabled houses of Amsterdam, bicycle in foreground': (52.3676, 4.9041), 'Sagrada Familia and Gothic Quarter in Barcelona, intricate details': (41.3851, 2.1734), 'Gondolas on the Grand Canal in Venice, charming and romantic': (45.4408, 12.3155), 'Charles Bridge and Prague Castle in Prague, misty morning': (50.0755, 14.4378), 'Bamboo groves and ancient temples in Kyoto, tranquil scene': (35.0116, 135.7681), 'Hagia Sophia and Blue Mosque in Istanbul, cultural blend': (41.0082, 28.9784), 'Golden Gate Bridge and Alcatraz in San Francisco, foggy day': (37.7749, -122.4194), 'Hollywood Sign and Griffith Observatory overlooking Los Angeles': (34.0522, -118.2437), "Cloud Gate 'The Bean' in Millennium Park, Chicago, urban art": (41.8781, -87.6298), 'CN Tower and downtown Toronto skyline at night, reflective surfaces': (43.6532, -79.3832), 'Stanley Park and mountains surrounding Vancouver, natural beauty': (49.2827, -123.1207), 'Zocalo square and Metropolitan Cathedral in Mexico City, bustling street': (19.4326, -99.1332), 'Obelisco and colorful La Boca in Buenos Aires, tango dancers': (-34.6037, -58.3816), 'Andes mountains backdrop for Santiago, modern city view': (-33.4489, -70.6693), 'Plaza Mayor and historical center of Lima, colonial architecture': (-12.0464, -77.0428), 'Ibirapuera Park and skyscrapers of São Paulo, green urban space': (-23.5505, -46.6333), 'Table Mountain overlooking Cape Town, vibrant waterfront': (-33.9249, 18.4241), 'Nelson Mandela Square and bustling streets of Johannesburg': (-26.2041, 28.0473), 'Giraffe Manor and wildlife near Nairobi National Park, unique experience': (-1.2921, 36.8219), 'Jemaa el-Fna square in Marrakech at dusk, snake charmers and storytellers': (31.6295, -7.9811), 'Dome of the Rock and Old City of Jerusalem, ancient and holy sites': (31.7683, 35.2137), 'Gyeongbokgung Palace in Seoul, traditional Korean architecture with modern skyline': (37.5665, 126.978), 'Floating markets and temples in Bangkok, vibrant street life': (13.7563, 100.5018), 'Hoan Kiem Lake and Old Quarter of Hanoi, serene and bustling': (21.0278, 105.8342), 'Petronas Twin Towers in Kuala Lumpur, impressive night view': (3.139, 101.6869), 'Intramuros historical walls and modern Makati skyline in Manila': (14.5995, 120.9842), 'National Monument (Monas) and street food vendors in Jakarta': (-6.2088, 106.8456), 'Gateway of India and bustling markets of Mumbai, colonial and local blend': (19.076, 72.8777), 'India Gate and Red Fort in Delhi, historical and vibrant': (28.7041, 77.1025), 'Boudhanath Stupa and narrow alleys of Kathmandu, spiritual atmosphere': (27.7172, 85.324), 'Potala Palace in Lhasa, serene and majestic, high altitude': (29.6525, 91.1322), 'The Bund and Pudong skyline in Shanghai, futuristic and historical contrast': (31.2304, 121.4737), 'Victoria Harbour and Hong Kong skyline at night, vibrant lights': (22.3193, 114.1694), 'Taipei 101 and night market scene in Taipei, lively street': (25.033, 121.5654), 'Sky Tower and Waitematā Harbour in Auckland, sailing boats': (-36.8485, 174.7633), 'Flinders Street Station and laneways of Melbourne, street art and cafes': (-37.8136, 144.9631), 'Story Bridge and South Bank Parklands in Brisbane, relaxed river city': (-27.4698, 153.0251), 'Kings Park overlooking Perth skyline and Swan River, natural beauty': (-31.9505, 115.8605), 'Waikiki Beach and Diamond Head in Honolulu, tropical paradise': (21.3069, -157.8583), 'Hallgrímskirkja church and colorful houses in Reykjavik, unique architecture': (64.9631, -19.0208), "Ha'penny Bridge and River Liffey in Dublin, lively pub scene": (53.3498, -6.2603), 'Edinburgh Castle and Royal Mile, historic Scottish charm': (55.9533, -3.1883), 'Northern Quarter and industrial heritage of Manchester, vibrant nightlife': (53.4808, -2.2426), 'Liverpool Cathedral and Beatles landmarks, cultural significance': (53.4084, -2.9916), 'Cardiff Castle and Millennium Centre, modern and historic blend': (51.4816, -3.1791), 'Oslo Opera House and Aker Brygge waterfront, modern Scandinavian design': (59.9139, 10.7522), 'Gamla Stan (Old Town) and Royal Palace in Stockholm, charming alleys': (59.3293, 18.0686), 'Nyhavn harbor and colorful houses in Copenhagen, relaxed atmosphere': (55.6761, 12.5683), 'Helsinki Cathedral and Market Square, classical architecture': (60.1695, 24.9354), 'Hermitage Museum and Neva River in St. Petersburg, grand architecture': (59.9343, 30.3351), 'Schönbrunn Palace and classical music halls in Vienna, imperial beauty': (48.2082, 16.3738), 'Parliament Building and Chain Bridge over Danube in Budapest, night lights': (47.4979, 19.0402), 'Old Town Market Square and Palace of Culture and Science in Warsaw, resilience': (52.2297, 21.0122), 'Belém Tower and Jerónimos Monastery in Lisbon, Tagus River views': (38.7223, -9.1393), 'Plaza Mayor and Royal Palace in Madrid, vibrant city life': (40.4168, -3.7038), 'Acropolis and Parthenon in Athens, ancient Greek ruins': (37.9838, 23.7275), 'Bibliotheca Alexandrina and coastal views in Alexandria': (31.2001, 29.9187), 'Hassan II Mosque and Art Deco architecture in Casablanca': (33.5731, -7.5898), 'Monument of the African Renaissance and coastline in Dakar': (14.6928, -17.4467), 'Independence Square and bustling markets of Accra, West African vibrancy': (5.6037, -0.187), 'Lekki Conservation Centre and markets of Lagos, dynamic African city': (6.5244, 3.3792), 'Kingdom Centre Tower and desert modernism in Riyadh': (24.7136, 46.6818), "King Fahd's Fountain and Corniche in Jeddah, Red Sea views": (21.4858, 39.1925), 'Museum of Islamic Art and modern skyline of Doha, waterfront view': (25.2854, 51.531), 'Sheikh Zayed Grand Mosque and Corniche in Abu Dhabi, majestic architecture': (24.4539, 54.3773), 'Roman Theater and ancient ruins in Amman, historical overlay': (31.9539, 35.9106), 'Raouché Rocks and vibrant waterfront of Beirut, Mediterranean charm': (33.8938, 35.5018), 'Umayyad Mosque and Old City of Damascus, historical depth': (33.5104, 36.2784), 'Al-Shaheed Monument and Tigris River in Baghdad, resilience': (33.3152, 44.3661), 'Azadi Tower and Milad Tower in Tehran, modern and symbolic': (35.6892, 51.389), 'Frere Hall and Clifton Beach in Karachi, bustling port city': (24.8608, 67.0099), 'Badshahi Mosque and Lahore Fort, Mughal architecture': (31.5498, 74.3436), 'Lalbagh Fort and rickshaws in Dhaka, vibrant and crowded': (23.8103, 90.4125), 'Howrah Bridge and Victoria Memorial in Kolkata, colonial grandeur': (22.5726, 88.3639), 'Marina Beach and Kapaleeshwarar Temple in Chennai, coastal and spiritual': (13.0827, 80.2707), 'Vidhana Soudha and Cubbon Park in Bangalore, garden city with modern tech': (12.9716, 77.5946), 'Charminar and Golconda Fort in Hyderabad, historic monuments': (17.385, 78.4867), 'Osaka Castle and Dotonbori entertainment district in Osaka, food and lights': (34.6937, 135.5022), 'Canal City Hakata and ancient temples in Fukuoka, modern and traditional': (33.5904, 130.4017), 'Nagoya Castle and bustling Sakae district in Nagoya, industrial and vibrant': (35.1815, 136.9064), 'Gamcheon Culture Village and Haeundae Beach in Busan, colorful and coastal': (35.1796, 129.0756), 'Canton Tower and Pearl River in Guangzhou, modern cityscape': (23.1291, 113.2644), 'Ping An Finance Center and vibrant tech hub of Shenzhen, futuristic city': (22.5428, 114.0579), 'Giant Panda Breeding Research Base and lively streets of Chengdu, unique culture': (30.5728, 104.0668), "Terracotta Army Museum and ancient city walls of Xi'an, historical wonders": (34.3416, 108.9398)}

def main():
    logging.info('**************************************')
    logging.info('**** Starting GeoCLIP experiments ****')
    logging.info('**************************************')
    parser = argparse.ArgumentParser(description="GeoCLIP main entry point")
    parser.add_argument('--data_dir', type=str, help='Path to input file or data')
    parser.add_argument('--dataset_size', type=int, default=100, help='Size of the dataset to generate')
    parser.add_argument('--images_class', type=str, default='city', help='Class of images to generate')
    args = parser.parse_args()
    logging.info(f'Configuration: {args}')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    if args.data_dir is not None:
        data_dir = args.data_dir
        logging.info(f'Using provided data directory: {data_dir}')

        image_dataset = ImageDataset('images', city_dict)
        image_loader = DataLoader(image_dataset, batch_size=8, shuffle=True)
        logging.info('Image dataset created successfully!')
    else:
        prompt = (
            f"Generate a valid Python dictionary (not code block) with exactly {args.dataset_size} entries, where the keys are prompts to generate {args.images_class} images, "
            f"and the value is the tuple of floats GPS location of the {args.images_class}. Only output the dictionary."
        )
        print('Generating prompts...')
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        print(GEMINI_API_KEY)
        image_dict = generate_dictionary(prompt, API_KEY=GEMINI_API_KEY)
        print('Prompts generated successfully!')
        prompts = list(image_dict.keys())
        logging.info('Generated prompts:')
        for prompt in prompts:
            logging.info(prompt)

        logging.info('Starting generating images...')
        generate_images(prompts, device=device)
        logging.info('Images generated successfully!')

        image_dataset = ImageDataset('images', image_dict)
        image_loader = DataLoader(image_dataset, batch_size=8, shuffle=True)
        logging.info('Image dataset created successfully!')

    model = GeoCLIP().to("cuda:1")
    print("===========================")
    print("GeoCLIP has been loaded! 🎉")
    print("===========================")
    logging.info('GoeCLIP model loaded successfully!')

    results = eval.eval_images(image_loader, model, device=device)
    logging.info(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()