import sys
from pathlib import Path
import random
import requests
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, auth
from rich.console import Console

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# ─── Vietnamese Names ───
first_names = [
    'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan', 'Vũ', 'Võ', 'Đặng',
    'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý', 'Đinh', 'Trịnh', 'Tô', 'La',
    'Mai', 'Tạ', 'Châu', 'Tăng', 'Lâm', 'Chu', 'Thái', 'Tiêu', 'Quách', 'Hà'
]

fallback_last_names = [
    'Gia Huy', 'Minh Khang', 'Khánh An', 'Bảo Ngọc', 'Hải Đăng',
    'Nhật Minh', 'Quỳnh Anh', 'Phương Linh', 'Yến Nhi', 'Đức Thịnh',
    'Hoàng Nam', 'Tuấn Kiệt', 'Thanh Trúc', 'Diễm My', 'Ngọc Mai',
    'Kim Ngân', 'Hà My', 'Khôi Nguyên', 'Minh Quân', 'Tiến Đạt',
    'Vân Anh', 'Thiên Ân', 'Bảo Châu', 'Quang Huy', 'Mỹ Duyên',
    'Anh Thư', 'Tường Vy', 'Hữu Phước', 'Gia Linh', 'Đức Anh'
]

# ─── Fallback Games ───
fallback_games = [
    'Liên Quân Mobile', 'PUBG Mobile', 'Free Fire', 'Tốc Chiến', 'Valorant',
    'League of Legends', 'Đấu Trường Chân Lý', 'CS2', 'Genshin Impact',
    'Honkai: Star Rail', 'Zenless Zone Zero', 'Minecraft', 'Roblox',
    'FC Online', 'Naraka: Bladepoint', 'Apex Legends', 'Overwatch 2',
    'Warzone Mobile', 'Marvel Rivals'
]

# ─── Fallback Ranks ───
fallback_ranks = [
    'Gà Mờ', 'Tập Sự Truyền Thuyết', 'Chiến Binh Phèn', 'Thánh Né',
    'Quái vật cân team', 'Trùm Cuối', 'Thượng Đế AFK',
    'Đồng', 'Bạc', 'Vàng', 'Bạch kim', 'Kim cương',
    'Cao thủ', 'Đại cao thủ', 'Thách đấu'
]

# ─── Locations GSO weighted ───
locations = [
    {'city': 'Hà Nội', 'lat': 21.0278, 'lng': 105.8342},
    {'city': 'TP. Hồ Chí Minh', 'lat': 10.7769, 'lng': 106.7009},
    {'city': 'Đà Nẵng', 'lat': 16.0678, 'lng': 108.2208},
    {'city': 'Hải Phòng', 'lat': 20.8449, 'lng': 106.6881},
    {'city': 'Cần Thơ', 'lat': 10.0452, 'lng': 105.7469},
    {'city': 'Nha Trang', 'lat': 12.2388, 'lng': 109.1967},
    {'city': 'Huế', 'lat': 16.4637, 'lng': 107.5909},
    {'city': 'Đà Lạt', 'lat': 11.9404, 'lng': 108.4583},
    {'city': 'Quy Nhơn', 'lat': 13.7820, 'lng': 109.2197},
    {'city': 'Vũng Tàu', 'lat': 10.4114, 'lng': 107.1362},
    {'city': 'Buôn Ma Thuột', 'lat': 12.6667, 'lng': 108.0500},
    {'city': 'Thái Nguyên', 'lat': 21.5942, 'lng': 105.8482},
    {'city': 'Long Xuyên', 'lat': 10.3833, 'lng': 105.4333},
    {'city': 'Rạch Giá', 'lat': 10.0125, 'lng': 105.0808},
]
location_weights = [93, 85, 35, 21, 18, 15, 12, 10, 9, 11, 8, 7, 6, 6]

def get_random_location_weighted():
    return random.choices(locations, weights=location_weights, k=1)[0]

def remove_vietnamese_tones(s):
    vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    latin_chars =      "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    trans_table = str.maketrans(vietnamese_chars, latin_chars)
    return s.translate(trans_table)

def generate_natural_bio():
    greetings = [
        'Hi mọi người!', 'Chào bạn nha,', 'Hello, mình là game thủ hệ vui vẻ.',
        'Xin chào các chiến hữu!', 'Tìm đồng đội hợp cạ đây.', 'Hi, mong tìm được cạ cứng.',
        'Hello ace!', 'Vào đây để giao lưu kết bạn.'
    ]
    roles = [
        'Mình chuyên main support, gánh team bằng cả tấm lòng.',
        'Chuyên đi rừng gank dạo, cần tìm lane tốt.',
        'Main mid/adc thích chơi chủ động.',
        'Thích fill mọi vị trí, chơi gì cũng được.',
        'Thích chơi game bắn súng FPS, tay to gánh team.',
        'Học việc MOBA, mong được mọi người chỉ giáo.',
        'Chơi đủ thể loại từ chiến thuật đến nhập vai.',
        'Chuyên đi đường rồng gánh team.'
    ]
    online_times = [
        'Tối online sau 8h.', 'Rảnh giờ nào chơi giờ đó, rủ là đi.',
        'Cuối tuần cày cật lực, ngày thường chơi tối.',
        'Chỉ online đêm muộn 22h - 2h sáng.', 'Thời gian online linh hoạt.',
        'Chiều tối 18h - 21h hàng ngày.', 'Thường rảnh cuối tuần.'
    ]
    call_to_actions = [
        'Ai cùng chí hướng thì match cùng leo rank nhé!',
        'Không toxic, thua cùng chịu, thắng cùng vui. Match nha!',
        'Vui lòng match mình nếu muốn try hard.',
        'Chơi game xả stress, match nói chuyện vui vẻ.',
        'Ai rảnh inbox giao lưu nhé!', 'Duo hoặc lập team 5 đều ok, match đi.',
        'Match đi chờ chi!', 'Tìm cạ cứng leo rank nghiêm túc.'
    ]
    return f"{random.choice(greetings)} {random.choice(roles)} {random.choice(online_times)} {random.choice(call_to_actions)}"

def main():
    console.print("[bold cyan]=== STARTING HYBRID TEST USER UPDATE SYSTEM ===[/bold cyan]\n")

    # Connect to Firebase
    credentials_path = Path(os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase.json"))
    project_id = os.getenv("FIREBASE_PROJECT_ID", "gamenect-9bec0")
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {"projectId": project_id})
        console.print("[green]✓ Firebase initialized successfully.[/green]")
    
    db = firestore.client()

    # 1. Fetch names from GitHub
    boys_names = []
    girls_names = []
    try:
        boy_res = requests.get('https://raw.githubusercontent.com/duyet/vietnamese-namedb/master/boy.txt')
        if boy_res.status_code == 200:
            boys_names = [line.strip() for line in boy_res.text.split('\n') if line.strip()]
            console.print(f"[green]✓ Loaded {len(boys_names)} boy names from GitHub.[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to load boy names: {e}. Fallback will be used.[/yellow]")

    try:
        girl_res = requests.get('https://raw.githubusercontent.com/duyet/vietnamese-namedb/master/girl.txt')
        if girl_res.status_code == 200:
            girls_names = [line.strip() for line in girl_res.text.split('\n') if line.strip()]
            console.print(f"[green]✓ Loaded {len(girls_names)} girl names from GitHub.[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to load girl names: {e}. Fallback will be used.[/yellow]")

    # 2. Fetch games from RAWG API
    games_list = []
    try:
        api_key = '754a38d2419a4aee8924fd13b8193b0f'
        url = f'https://api.rawg.io/api/games?key={api_key}&page_size=40&ordering=-added'
        games_res = requests.get(url)
        if games_res.status_code == 200:
            data = games_res.json()
            games_list = [g['name'] for g in data.get('results', []) if g.get('name')]
            console.print(f"[green]✓ Loaded {len(games_list)} games from RAWG API.[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to load RAWG games: {e}. Fallback will be used.[/yellow]")

    # 3. Fetch ranks from Firestore configs
    ranks_list = []
    try:
        rank_doc = db.collection('configurations').document('rankOptions').get()
        if rank_doc.exists:
            data = rank_doc.to_dict()
            if data and 'vi' in data:
                ranks_list = list(data['vi'])
                console.print(f"[green]✓ Loaded {len(ranks_list)} ranks from Firestore config.[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to load ranks from Firestore: {e}. Fallback will be used.[/yellow]")

    # Query all test accounts
    console.print("\nQuerying test users from Firestore...")
    users_ref = db.collection('users').where('isTestAccount', '==', True)
    docs = list(users_ref.stream())
    console.print(f"[cyan]Found {len(docs)} test users in Firestore.[/cyan]\n")

    if not docs:
        console.print("[yellow]No test users found to update.[/yellow]")
        return

    # Process updates
    updated_count = 0
    for idx, doc in enumerate(docs, start=1):
        uid = doc.id
        data = doc.to_dict()
        
        gender = data.get('gender', 'Nam')
        if gender not in ['Nam', 'Nữ', 'Khác']:
            gender = random.choice(['Nam', 'Nữ', 'Khác'])

        # Name selection
        ho = random.choice(first_names)
        if gender == 'Nam':
            ten = random.choice(boys_names) if boys_names else random.choice(fallback_last_names)
        elif gender == 'Nữ':
            ten = random.choice(girls_names) if girls_names else random.choice(fallback_last_names)
        else:
            pool = boys_names + girls_names if (boys_names or girls_names) else fallback_last_names
            ten = random.choice(pool)

        display_name = f"{ho} {ten}"
        username = f"{remove_vietnamese_tones(ho.lower())}{remove_vietnamese_tones(ten.lower())}{idx:03d}"
        
        if idx < 100:
            email = f"testuser{idx:04d}@gamenect.com"
        else:
            email = f"{username.replace(' ', '')}@gmail.com"

        # Weighted Location
        loc = get_random_location_weighted()
        
        # Natural Bio
        bio = generate_natural_bio()
        
        # Ranks
        rank_pool = ranks_list if ranks_list else fallback_ranks
        rank = random.choice(rank_pool)
        
        # Games
        games_pool = games_list if games_list else fallback_games
        num_games = random.randint(1, min(5, len(games_pool)))
        selected_games = random.sample(games_pool, num_games)

        # Update Auth
        try:
            auth.update_user(
                uid,
                email=email,
                display_name=display_name
            )
        except Exception as auth_err:
            console.print(f"[red]Error updating Auth for {uid} ({email}): {auth_err}. Attempting Firestore only.[/red]")

        # Update Firestore
        profile_data = {
            'email': email,
            'username': username,
            'displayName': display_name,
            'gender': gender,
            'bio': bio,
            'favoriteGames': selected_games,
            'rank': rank,
            'location': {
                'city': loc['city'],
                'latitude': loc['lat'],
                'longitude': loc['lng'],
                'updatedAt': firestore.SERVER_TIMESTAMP
            },
            'updatedAt': firestore.SERVER_TIMESTAMP
        }

        try:
            db.collection('users').document(uid).update(profile_data)
            updated_count += 1
            if idx % 50 == 0 or idx == len(docs):
                console.print(f"Progress: updated {idx}/{len(docs)} users... Current: {display_name} ({email}) | {loc['city']}")
        except Exception as fs_err:
            console.print(f"[red]Error updating Firestore for {uid}: {fs_err}[/red]")

    console.print(f"\n[bold green]=== SYSTEM COMPLETED SUCCESSFULLY. UPDATED {updated_count}/{len(docs)} TEST USERS ===[/bold green]")

if __name__ == "__main__":
    main()
