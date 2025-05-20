# Packages
import os
import csv
import json
import jwt
import hashlib
import time
import pytz
import app.models as models
from app.extensions import *
from flask_mail import Message
from datetime import datetime, timedelta
from jwt import ExpiredSignatureError, InvalidTokenError
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
from app.models import *
from flask import current_app
from google.generativeai import GenerativeModel
from flask import Blueprint, request, make_response, url_for, Response, jsonify
from sqlalchemy import func, desc
from datetime import datetime
from flask import Flask
from geopy.distance import geodesic

IST = pytz.timezone("Asia/Kolkata")

main = Blueprint( 'main', __name__)

load_dotenv()

s = URLSafeTimedSerializer(os.getenv('SECRET_KEY'))

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
BASE_URL = os.getenv('BASE_FRONTEND_URL')

# Authenctication
def verify_token():
    token = request.cookies.get('cookie')

    if not token:
        return {'message': 'Token not found in cookies'}, 401

    try:
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return {'user': decoded['user'], 'role': decoded['role']}, 200

    except ExpiredSignatureError:
        return {'message': 'Token expired'}, 401
    except InvalidTokenError:
        return {'message': 'Invalid token'}, 401
    except Exception as e:
        return {'message': f'Token verification failed: {str(e)}'}, 400
    

def send_confirm_mail(email):
    try:
        token = s.dumps(email, salt='email-confirm')
        msg = Message('Confirm Your Email, for SmartPDS', sender=f'{os.environ.get('MAIL_USERNAME')}', recipients=[email])
        link = url_for('main.confirm_email', token=token, _external=True)
        msg.body = f'Click here {link} to confirm your account\n\nRegards\nSmartPDS'
        mail.send(msg)
        return True
    
    except Exception as e_msg:
        print(e_msg)
        return False
 

def send_stockout_email(to, subject, body):
    msg = Message(subject, sender=f'{os.environ.get('MAIL_USERNAME')}', recipients=[to])
    msg.body = body
    mail.send(msg)


def get_user_id(email: str)-> int:
    user = User.query.filter_by(email=email).first_or_404()
    print(user.user_id)
    return user.user_id


def run_auto_stock_out_job(app, shop_id):
    with app.app_context():
        auto_stock_out(shop_id=shop_id)
        
  
@main.route('/signup',methods=['POST'])
def signup():
    data = request.get_json(force=True)
    given_email = data.get('email')
    given_password = data.get('password')
    role = data.get('role', 'public')

    # if not email.endswith('gov.in'):
    #     return make_response({'message': 'Invalid email domain'}, 400)
    
    existing_user = User.query.filter_by(email=given_email).first()

    if existing_user and not check_password_hash(existing_user.password, given_password):
        return make_response({'message': 'Incorrect password'}, 400)
    
    if existing_user and check_password_hash(existing_user.password, given_password):

        if not existing_user.verified:

            success_mail = send_confirm_mail(given_email)
            if not success_mail:
                return make_response({"message": "Error sending mail"}, 500)
            
            return make_response(
                {'message': 'Verification email resent. Please check your email to verify your account'}, 200)
        
        return make_response({'message': 'Email already exists and verified'}, 400)

    hashed_password = generate_password_hash(given_password, method='pbkdf2:sha256')
    new_user = User(email=given_email, password=hashed_password, verified=False, role=role)

    db.session.add(new_user)
    db.session.commit()

    success_mail = send_confirm_mail(given_email)
    if not success_mail:
        return make_response({"message": "Error sending mail"}, 500)

    return make_response({'message': 'User created, please check your email to verify your account'}, 201)


@main.route('/signin', methods=['POST'])
def signin():
    data = request.get_json(force=True)
    given_email = data.get('email')
    given_password = data.get('password')

    user = User.query.filter_by(email=given_email).first()

    if not user or not check_password_hash(user.password, given_password):
        return make_response({'message': 'Invalid credentials'}, 400)

    if not user.verified:
        return make_response({'message': 'Email not verified'}, 400)
    
    payload = {
        'user': user.email,
        'exp': datetime.utcnow() + timedelta(minutes=30),
        'role': user.role 
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

    response = make_response({'message': 'Logged in successfully', 'token': token, 'role':user.role, 'email': user.email}, 200)

    return response


@main.route("/ask", methods=["POST"])
def ask_full_db():
    payload = request.get_json(force=True)
    query = payload["query"]
    result = ask_gemini_over_db(query)
    return make_response({"result":result}, 200)


@main.route('/verify_account/<token>', methods=['GET'])
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=300)

    except SignatureExpired:
        Response(json.dumps({'message': 'The token has expired'}), status=400)
        return "<center><h1>The token has expired</h1></center>"
    
    except BadSignature:
        Response(json.dumps({'message': 'Invalid token'}), status=400)
        return "<center><h1>Invalid Token</h1></center>"

    user = User.query.filter_by(email=email).first_or_404()
    if user.verified:
        Response( json.dumps({'message': 'Account already verified'}), status=400)
        return "<center><h1>Email verified successfully</h1></center>"

    user.verified = True
    db.session.commit()
    Response(json.dumps({'message': 'Email verified successfully'}))
    return f"<center><h1>Your email has been verified successfully</h1><p>Now, you can <a href='{BASE_URL}/login'>SignIn</a></p></center>"
    

@main.route('/add_item', methods=['POST'])
def add_item():
    data = request.get_json(force=True)
    item_name = data.get('name').lower() if data['name'] else None

    if not item_name:
        return make_response({'message': 'Item name is required'}, 400)

    if Item.query.filter_by(name=item_name).first():
        return make_response({'message': 'Item already exists'}, 400)

    item = Item(name=item_name)
    db.session.add(item)
    db.session.commit()
    return make_response({'message': 'Product added', 'item_id': item.id}, 200)


@main.route('/product_details', methods=['GET'])
def products_summary():
    item_counts = (
        db.session.query(
            Item.id,
            Item.name,
            func.sum(StockItem.remaining_quantity).label('total_remaining')
        )
        .join(StockItem, Item.id == StockItem.item_id)
        .group_by(Item.id)
        .all()
    )

    item_count_data = [
        {'item_id': item.id, 'item_name': item.name, 'total_remaining': int(item.total_remaining or 0)}
        for item in item_counts
    ]

    latest_items = (
        db.session.query(StockItem, Item)
        .join(Item, StockItem.item_id == Item.id)
        .order_by(StockItem.id.desc())
        .limit(5)
        .all()
    )

    latest_added = [
        {
            'stock_id': stock.id,
            'item_id': item.id,
            'item_name': item.name,
            'quantity': stock.total_quantity,
            'godown_id': stock.godown_id,
            'batch_id': stock.batch_id
        }
        for stock, item in latest_items
    ]

    requirements = (
        db.session.query(
            Item.id,
            Item.name,
            func.sum(ShopRequirement.required_quantity).label('total_needed')
        )
        .join(ShopRequirement, Item.id == ShopRequirement.item_id)
        .group_by(Item.id)
        .having(func.sum(ShopRequirement.required_quantity) > 0)
        .all()
    )

    needed_items = [
        {'item_id': item.id, 'item_name': item.name, 'total_needed': int(item.total_needed)}
        for item in requirements
    ]

    return make_response({
        'item_counts': item_count_data,
        'latest_added': latest_added,
        'needed_items': needed_items
    }, 200)


@main.route('/add_godown', methods=['POST'])
def add_godown():
    data = request.get_json(force=True)
    name = data.get('name')
    location = data.get('location')

    if not name or not location:
        return make_response({'error': 'Name and location are required'}, 400)

    godown = Godown(name=name, location=location)
    db.session.add(godown)
    db.session.commit()
    return make_response({'message': 'Godown added', 'godown_id': godown.id}, 200)


@main.route('/add_shop', methods=['POST'])
def add_pds_shop():
    data = request.get_json(force=True)
    name = data.get('name')
    location = data.get('location')
    owner = data.get('owner')
    waypoints = data.get('waypoints', [])
    destination = data.get('destination')

    if not name or not location or not owner or not waypoints or not destination:
        return make_response({'message': 'Missing required Fields!'}, 400)

    shop = PDSShop(name=name, location=location, owner=owner, waypoints=waypoints, destination=destination)
    db.session.add(shop)
    db.session.commit()

    user = User.query.filter_by(id=owner).first()

    if user:
        subject = f"Shop Registered - {name}"
        body = f"Dear User,\n\nPDSShop has been registered under your name as owner\nShop Details: name:{shop.name}, location:({shop.location}).\n\nRegards,\nSmardPDS"

        if user.email:
            send_stockout_email(
                to=user.email,
                subject=subject,
                body=body
            )
    return make_response({'message': 'PDS shop added!', 'shop_id': shop.id}, 200)


@main.route('/add_shop_requirement', methods=['POST'])
def add_shop_requirement():
    data = request.get_json(force=True)
    shop_id = data.get('shop_id')
    requirements = data.get('requirements')  # List of {'item_id': int, 'required_quantity': int}

    if not shop_id or not requirements:
        return make_response({'message': 'Missing requirements!'}, 400)

    for req in requirements:
        item_id = req.get('item_id')
        quantity = req.get('required_quantity')
        if not item_id or not quantity:
            continue

        existing = ShopRequirement.query.filter_by(shop_id=shop_id, item_id=item_id).first()
        if existing:
            existing.required_quantity = quantity  
        else:
            db.session.add(ShopRequirement(shop_id=shop_id, item_id=item_id, required_quantity=quantity))

    db.session.commit()
    return make_response({'message': 'Requirements added or updated'}, 200)


@main.route('/stock_allocations/<int:shop_id>', methods=['GET'])
def get_stock_allocations(shop_id):
    allocations = StockAllocation.query.filter_by(shop_id=shop_id).all()
    data = []
    for a in allocations:
        data.append({
            'stock_item_id': a.stock_item_id,
            'quantity_allocated': a.quantity_allocated,
            'allocated_at': a.allocated_at.isoformat()
        })
    return make_response(data, 200)


@main.route('/stock_in', methods=['POST'])
def stock_in():
    data = request.get_json(force=True)
    items = data['items']
    godown_id = data['godown_id']
    all_pack_codes = []

    if not items or not godown_id:
        return make_response({'message': 'Godown and Items are required'}, 400)

    godown_id = int(godown_id)

    # Create one batch for all items
    new_batch = Batch(arrival_date=datetime.utcnow())
    db.session.add(new_batch)
    db.session.flush()  # Get batch.id

    for item in items:
        item_id = int(item['item_id'])
        total_quantity = int(item['quantity'])
        num_of_packs = int(item['num_of_packs'])

        stock_item = StockItem(
            batch_id=new_batch.id,
            godown_id=godown_id,
            item_id=item_id,
            total_quantity=total_quantity,
            remaining_quantity=total_quantity,
            num_of_packs=num_of_packs,
            current_num_of_packs=num_of_packs,
            pack_codes=None  # Will update below
        )
        db.session.add(stock_item)
        db.session.flush()  # Get stock_item.id

        # Generate pack codes
        codes = []
        for i in range(1, num_of_packs + 1):
            code = f"{new_batch.id}-{stock_item.id}-{item_id}-{str(i).zfill(3)}"
            codes.append(code)
            all_pack_codes.append([code])

        stock_item.pack_codes = codes[-1]  # Save last code (you can store all if required)

    # Blockchain entry (single entry for the entire batch)
    last_block = Blockchain.query.order_by(Blockchain.id.desc()).first()
    previous_hash = last_block.hash if last_block else '0'

    block_data = {
        'action': 'stock_in',
        'batch_id': new_batch.id,
        'godown_id': godown_id,
        'items': [
            {
                'item_id': int(item['item_id']),
                'quantity': int(item['quantity']),
                'num_packs': int(item['num_of_packs'])
            } for item in items
        ]
    }

    index = (last_block.index + 1) if last_block else 1
    timestamp = datetime.utcnow().isoformat()
    data_string = json.dumps(block_data, sort_keys=True)
    block_hash = hashlib.sha256(f"{index}{timestamp}{data_string}{previous_hash}".encode()).hexdigest()

    block = Blockchain(index=index, timestamp=timestamp, data=data_string, previous_hash=previous_hash, hash=block_hash)
    db.session.add(block)

    # Save all codes to CSV
    documents_path = os.path.expanduser("~/Documents")
    target_folder = os.path.join(documents_path, "generated_codes")
    os.makedirs(target_folder, exist_ok=True)
    filename = f"stockin_codes_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
    filepath = os.path.join(target_folder, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pack Code"])
        writer.writerows(all_pack_codes)

    db.session.commit()
    return make_response({'message': 'Stock added under one batch.\n Check Documents folder for generated pack codes', 'file': filepath}, 200)


@main.route('/schedule_stock_out', methods=['POST'])
def schedule_stock_out():
    data = request.get_json(force=True)
    shop_id = data.get('shop_id')
    run_date_str = data.get('run_date')
    frequency = data.get('frequency', 'daily')

    try:
        naive_dt = datetime.fromisoformat(run_date_str)
        run_date = IST.localize(naive_dt)
    except ValueError:
        return make_response({'error': 'Invalid date format'}, 400)

    job_id = f"stockout_shop_{shop_id}"

    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    ScheduledJob.query.filter_by(shop_id=shop_id).delete()
    db.session.commit()

    scheduled = ScheduledJob(shop_id=shop_id, run_date=run_date, frequency=frequency)
    db.session.add(scheduled)
    db.session.commit()

    app = current_app._get_current_object()

    if frequency == 'daily':
        scheduler.add_job(
            func=run_auto_stock_out_job,
            trigger='interval',
            days=1,
            start_date=run_date,
            args=[app, shop_id],
            id=job_id,
            replace_existing=True,
            timezone=IST,
        )
    elif frequency == 'weekly':
        scheduler.add_job(
            func=run_auto_stock_out_job,
            trigger='interval',
            weeks=1,
            start_date=run_date,
            args=[app, shop_id],
            id=job_id,
            replace_existing=True,
            timezone=IST,
        )
    elif frequency == 'monthly':
        scheduler.add_job(
            func=run_auto_stock_out_job,
            trigger='cron',
            day=run_date.day,
            hour=run_date.hour,
            minute=run_date.minute,
            args=[app, shop_id],
            id=job_id,
            replace_existing=True,
            timezone=IST,
        )
    else:
        return make_response({'message': 'Invalid frequency'}, 400)

    return make_response({
        'message': f'Stock out for shop {shop_id} scheduled every {frequency} starting from {run_date.strftime("%Y-%m-%d %H:%M:%S")}'
    }, 200)


@main.route('/stock_out/<int:shop_id>', methods=['POST'])
def auto_stock_out(shop_id):
    requirements = ShopRequirement.query.filter_by(shop_id=shop_id).all()
    response_data = []

    shop = PDSShop.query.get(shop_id)
    vehicle = Transport.query.filter_by(status="at_inventory").first()

    if not vehicle:
        return make_response({"message": "No available vehicles"}, 500)

    vehicle.status = "on_the_way"
    vehicle.destination_shop_id = shop.id
    vehicle.waypoints = shop.waypoints

    for req in requirements:
        item_id = req.item_id
        quantity_needed = req.required_quantity
        remaining = quantity_needed

        item = Item.query.get(item_id)
        item_name = item.name if item else f"Item ID {item_id}"

        stock_items = StockItem.query.filter_by(item_id=item_id).filter(StockItem.remaining_quantity > 0).order_by(StockItem.batch_id).all()

        packs_taken_total = 0
        pack_codes_allocated = []

        for stock in stock_items:
            if remaining <= 0:
                break

            take = min(stock.remaining_quantity, remaining)
            packs_ratio = take / stock.total_quantity
            packs_to_take = int(round(stock.num_of_packs * packs_ratio))

            # Default to 0 if missing
            packs_to_take = min(packs_to_take, stock.current_num_of_packs or 0)
            if packs_to_take == 0:
                continue

            # Handle code generation from pack_codes (last code logic)
            last_code = stock.pack_codes
            if not last_code:
                continue  # no codes left

            try:
                batch_id, stock_id, item_id_str, last_number_str = last_code.split('-')
                last_number = int(last_number_str)
                prefix = f"{batch_id}-{stock_id}-{item_id_str}"
                new_last_number = last_number - packs_to_take

                # Generate allocated code range
                allocated_codes = [
                    f"{prefix}-{str(i).zfill(3)}"
                    for i in range(new_last_number + 1, last_number + 1)
                ]
                pack_codes_allocated.extend(allocated_codes)

                # Update remaining last code
                if new_last_number > 0:
                    stock.pack_codes = f"{prefix}-{str(new_last_number).zfill(3)}"
                else:
                    stock.pack_codes = None  # no packs left
            except Exception as e:
                print("Code parse error:", e)
                continue

            stock.remaining_quantity -= take
            stock.current_num_of_packs -= packs_to_take
            stock.current_num_of_packs = max(0, stock.current_num_of_packs)

            if stock.remaining_quantity == 0:
                stock.departure_date = datetime.utcnow()
                stock.num_of_packs = 0

            allocation = StockAllocation(
                stock_item_id=stock.id,
                shop_id=shop_id,
                quantity_allocated=take,
                packs_allocated=packs_to_take,
                allocated_pack_codes=allocated_codes
            )
            db.session.add(allocation)

            # Blockchain entry
            last_block = Blockchain.query.order_by(Blockchain.id.desc()).first()
            previous_hash = last_block.hash if last_block else '0'

            block_data = {
                'action': 'stock_out',
                'item_id': item_id,
                'quantity': take,
                'batch_id': stock.batch_id,
                'godown_id': stock.godown_id,
                'shop_id': shop_id,
                'pack_codes': allocated_codes
            }

            index = (last_block.index + 1) if last_block else 1
            timestamp = datetime.utcnow().isoformat()
            data_string = json.dumps(block_data, sort_keys=True)
            block_hash = hashlib.sha256(f"{index}{timestamp}{data_string}{previous_hash}".encode()).hexdigest()

            block = Blockchain(index=index, timestamp=timestamp, data=data_string, previous_hash=previous_hash, hash=block_hash)
            db.session.add(block)

            packs_taken_total += packs_to_take
            remaining -= take

        from_code = pack_codes_allocated[0] if pack_codes_allocated else None
        to_code = pack_codes_allocated[-1] if pack_codes_allocated else None

        response_data.append({
            'item': item_name,
            'item_id': item_id,
            'required_quantity': quantity_needed,
            'fulfilled': quantity_needed - remaining,
            'packs_sent': packs_taken_total,
            'status': 'partial' if remaining > 0 else 'fulfilled',
            'pack_code_range': f"{from_code} to {to_code}" if from_code and to_code else "N/A",
        })

    db.session.commit()

    user = User.query.filter_by(id=shop.owner).first()

    if user:
        subject = f"Stock Out Notification - {shop.name}"
        body = f"Dear User,\n\nStock has just been allocated to your PDS shop: {shop.name} ({shop.location}).\n\nDetails:\n"
        for r in response_data:
            body += f"- {r['item']}: {r['fulfilled']} out of {r['required_quantity']} ({r['status'].capitalize()}), Pack Codes: {r['pack_code_range']}\n"

        body += f"\nPlease prepare for delivery and confirm the pack labels.\nVechile No:{vehicle.vehicle_id}\n\nRegards,\nSmartPDS"

        if user.email:
            send_stockout_email(
                to=user.email,
                subject=subject,
                body=body
            )

    return make_response({
        'message': 'Stock out completed',
        'data': response_data,
        "vechile_id": vehicle.vehicle_id,
        "t_id": vehicle.tid
    }, 200)



    data = request.get_json(force=True)
    shop_id = data.get('shop_id')
    run_date_str = data.get('run_date')  # ISO format: '2025-05-01T12:00:00'
    frequency = data.get('frequency', 'daily')  # 'daily', 'weekly', 'monthly'

    try:
        run_date = datetime.fromisoformat(run_date_str)
    except ValueError:
        return make_response({'error': 'Invalid date format'}, 400)

    scheduled = ScheduledJob(shop_id=shop_id, run_date=run_date, frequency=frequency)
    db.session.add(scheduled)
    db.session.commit()

    job_id = f"stockout_shop_{shop_id}_{scheduled.id}"

    if frequency == 'daily':
        scheduler.add_job(
            func=run_auto_stock_out_job,
            trigger='interval',
            days=1,
            start_date=run_date,
            args=[shop_id],
            id=job_id,
            replace_existing=True
        )
    elif frequency == 'weekly':
        scheduler.add_job(
            func=run_auto_stock_out_job,
            trigger='interval',
            weeks=1,
            start_date=run_date,
            args=[shop_id],
            id=job_id,
            replace_existing=True
        )
    elif frequency == 'monthly':
        scheduler.add_job(
            func=run_auto_stock_out_job,
            trigger='cron',
            day=run_date.day,
            hour=run_date.hour,
            minute=run_date.minute,
            args=[shop_id],
            id=job_id,
            replace_existing=True
        )
    else:
        return make_response({'message': 'Invalid frequency'}, 400)

    return make_response({'message': f'Stock out scheduled for shop {shop_id} every {frequency} starting from {run_date}'}, 200)



    data = request.get_json(force=True)
    shop_id = data.get('shop_id')
    run_date_str = data.get('run_date')
    frequency = data.get('frequency', 'daily')

    try:
        naive_dt = datetime.fromisoformat(run_date_str)
        run_date = IST.localize(naive_dt)  # Convert to Asia/Kolkata timezone-aware datetime
    except ValueError:
        return make_response({'error': 'Invalid date format'}, 400)

    # Save or use `run_date` as timezone-aware datetime
    scheduled = ScheduledJob(shop_id=shop_id, run_date=run_date, frequency=frequency)
    db.session.add(scheduled)
    db.session.commit()

    job_id = f"stockout_shop_{shop_id}_{scheduled.id}"

    # Schedule job with timezone-aware run_date
    scheduler.add_job(
        func=run_auto_stock_out_job,
        trigger='interval' if frequency in ['daily', 'weekly'] else 'cron',
        days=1 if frequency == 'daily' else None,
        weeks=1 if frequency == 'weekly' else None,
        day=run_date.day if frequency == 'monthly' else None,
        hour=run_date.hour,
        minute=run_date.minute,
        start_date=run_date,
        timezone=IST,
        args=[shop_id],
        id=job_id,
        replace_existing=True
    )

    return make_response({'message': f'Stock out scheduled for shop {shop_id} every {frequency} from {run_date}'}, 200)


def get_all_model_classes():
    return [
        cls for name, cls in vars(models).items()
        if isinstance(cls, type) and issubclass(cls, db.Model) and cls != db.Model
    ]


def serialize_instance(obj):
    result = {}
    for column in obj.__table__.columns:
        value = getattr(obj, column.name)
        if isinstance(value, datetime):
            result[column.name] = value.isoformat()
        else:
            result[column.name] = value
    return result


def get_all_model_data(limit=20):
    model_classes = get_all_model_classes()
    db_data = {}

    for cls in model_classes:
        if cls.__name__.lower() in ['user', 'blockchain']:
            continue
        name = cls.__name__.lower() + "s"  # pluralize
        rows = cls.query.limit(limit).all()
        db_data[name] = [serialize_instance(r) for r in rows]

    return db_data


def ask_gemini_over_db(question):
    db_data = get_all_model_data()
    json_data = json.dumps(db_data, indent=2)
    model = GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a smart assistant working for the Public Distribution System (PDS) Inventory Management System.

    Your job is to assist registered inventory officials and PDS shop representatives by providing helpful, 
    user-friendly answers based on real-time supply and stock data.

    Strict Privacy Rules — DO NOT DO THE FOLLOWING:

        Never reveal or describe internal database structure.

        Never expose table names, column names, or code-level implementation.

        Never mention how data is stored, retrieved, or processed.

        Never reveal or reference backend logic, APIs, or storage details.

    What You Know:

        Stocks are delivered as batches from registered godowns to registered PDS shops.

        Deliveries are scheduled in advance and tracked.

        You are given access to interpreted live data.

    Now, using the following data:

    {json_data}

    Answer the question below in a friendly, clear, and non-technical way:

    {question}
    """

    response = model.generate_content(prompt)
    return response.text


@main.route('/dashboard', methods=['GET'])
def dashboard():
    total_shops = PDSShop.query.count()
    total_items = Item.query.count()
    total_users = User.query.count()
    total_stocks = StockItem.query.count()
    total_allocations = StockAllocation.query.count()
    total_jobs = ScheduledJob.query.count()

    total_quantity_in = db.session.query(db.func.sum(StockItem.total_quantity)).scalar() or 0
    total_quantity_remaining = db.session.query(db.func.sum(StockItem.remaining_quantity)).scalar() or 0
    total_quantity_out = total_quantity_in - total_quantity_remaining

    item_stats = []
    items = Item.query.all()
    for item in items:
        total_quantity = db.session.query(db.func.sum(StockItem.total_quantity)).filter_by(item_id=item.id).scalar() or 0
        remaining_quantity = db.session.query(db.func.sum(StockItem.remaining_quantity)).filter_by(item_id=item.id).scalar() or 0
        remaining_packs = db.session.query(db.func.sum(StockItem.current_num_of_packs)).filter_by(item_id=item.id).scalar() or 0

        status = "Low Stock" if total_quantity > 0 and remaining_quantity / total_quantity < 0.2 else "Normal"

        item_stats.append({
            "item_id": item.id,
            "item_name": item.name,
            "remaining_quantity": remaining_quantity,
            "remaining_packs": remaining_packs or 0,
            "status": status
        })
    
    total_capacity = 10000
    inventory_fill_percentage = (total_quantity_remaining / total_capacity) * 100 if total_capacity else 0

    return make_response({
        "summary": {
            "total_shops": total_shops,
            "total_items": total_items,
            "total_users": total_users,
            "total_stocks": total_stocks,
            "total_allocations": total_allocations,
            "total_scheduled_jobs": total_jobs
        },
        "stock_status": {
            "total_quantity_in": total_quantity_in,
            "total_quantity_remaining": total_quantity_remaining,
            "total_quantity_out": total_quantity_out
        },
        "item_details": item_stats,
        "total_capacity": '10,00,000',
        "inventory_fill_percentage": round(inventory_fill_percentage)
    }, 200)


@main.route("/item_quantity_summary", methods=["GET"])
def item_quantity_summary():
    items = Item.query.all()
    response = []

    for item in items:
        # Total required quantity from all shop requirements
        total_required = (
            db.session.query(db.func.sum(ShopRequirement.required_quantity))
            .filter(ShopRequirement.item_id == item.id)
            .scalar()
        ) or 0

        # Stock items for this item
        stock_items = StockItem.query.filter_by(item_id=item.id).all()

        # Allocated quantity from stock allocations
        allocated_quantity = 0
        for stock_item in stock_items:
            allocations = StockAllocation.query.filter_by(stock_item_id=stock_item.id).all()
            allocated_quantity += sum(alloc.quantity_allocated for alloc in allocations)

        # Remaining quantity from all stock items of this item
        remaining_quantity = sum(stock.remaining_quantity for stock in stock_items)

        response.append({
            "item_name": item.name,
            "required_quantity": total_required,
            "allocated_quantity": allocated_quantity,
            "remaining_quantity": remaining_quantity
        })

    return make_response({"body": response}, 200)


@main.route('/pds_shops_details', methods=['GET'])
def pds_shops_with_requirements():
    shops = PDSShop.query.all()
    result = []

    for shop in shops:
        shop_data = {
            "shop_id": shop.id if shop.id else None,
            "shop_name": shop.name if shop.name else None,
            "location": shop.location if shop.location else None,
            "owner": User.query.filter_by(id=shop.owner).first().email if shop.owner else None,
            "requirements": []
        }

        requirements = ShopRequirement.query.filter_by(shop_id=shop.id).all()
        
        for req in requirements:
            item = Item.query.get(req.item_id)

            shop_data["requirements"].append({
                "requirement_id": req.id,
                "item_name": item.name if item else "Unknown",
                "required_quantity": req.required_quantity
            })

        result.append(shop_data)
        
    return make_response({"body": result}, 200)


@main.route('/godowns_details', methods=['GET'])
def get_godowns():
    godowns = Godown.query.all()
    result = []

    for godown in godowns:
        stock_items = StockItem.query.filter_by(godown_id=godown.id).all()
        total_quantity = sum(item.total_quantity for item in stock_items)
        remaining_quantity = sum(item.remaining_quantity for item in stock_items)
        total_packs = sum(item.num_of_packs for item in stock_items)
        remaining_packs = sum(item.current_num_of_packs for item in stock_items)

        result.append({
            "godown_id": godown.id,
            "name": godown.name,
            "location": godown.location,
            "total_quantity": total_quantity,
            "remaining_quantity": remaining_quantity,
            "total_packs": total_packs,
            "remaining_packs": remaining_packs
        })

    return make_response({"body": result}, 200)


@main.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify({'users': [{'id': u.id, 'name': u.email} for u in users]}), 200


@main.route('/items', methods=['GET'])
def get_items():
    items = Item.query.all()
    return jsonify([{'id': i.id, 'name': i.name.capitalize()} for i in items]), 200


@main.route('/all_shops', methods=['GET'])
def get_shops():
    shops = PDSShop.query.all()
    return jsonify({'shops': [{'id': s.id, 'name': s.name} for s in shops]}), 200


def send_email(subject, recipient, body):
    msg = Message(subject=subject, sender=os.environ.get('MAIL_USERNAME'), recipients=recipient)
    msg.body = body
    mail.send(msg)


@main.route('/owntracks', methods=['POST'])
def receive_owntracks():
    data = request.get_json(force=True)
    tid = data.get('tid')
    lat = data.get('lat')
    lon = data.get('lon')

    if not tid or lat is None or lon is None:
        return {"status": "error", "message": "Missing tid, lat, or lon"}, 400

    # Save or update latest location in the database
    location = VehicleLocation.query.get(tid)
    if not location:
        location = VehicleLocation(tid=tid, lat=lat, lng=lon, timestamp=time.time())
        db.session.add(location)
    else:
        location.lat = lat
        location.lng = lon
        location.timestamp = time.time()

    db.session.commit()
    print(f"Updated {tid}: {lat}, {lon}")
    return {"status": "ok"}, 200


@socketio.on("track_vehicle")
def handle_track_vehicle(tid):
    location = VehicleLocation.query.get(tid)
    if not location:
        socketio.emit("vehicle_update", {"error": f"No location for TID {tid}"})
        return

    vehicle = Transport.query.filter_by(tid=tid).first()
    if not vehicle:
        socketio.emit("vehicle_update", {"error": f"Vehicle with TID {tid} not found"})
        return

    shop = PDSShop.query.get(vehicle.destination_shop_id)
    if not shop:
        socketio.emit("vehicle_update", {"error": f"Shop not found for TID {tid}"})
        return

    current_point = (location.lat, location.lng)
    destination_point = (shop.destination['lat'], shop.destination['lng'])

    # Distance calculation
    distance_to_destination = geodesic(current_point, destination_point).km

    # Deviation check
    deviation_alert = True
    for wp in shop.waypoints:
        if geodesic(current_point, (wp['lat'], wp['lng'])).km < 0.25:
            deviation_alert = False
            break

    admin = User.query.filter_by(role='admin').first()

    if deviation_alert and vehicle.status != "alert_sent":
        send_email(
            subject=f"Vehicle {vehicle.vehicle_id} deviated!",
            recipient=[admin.email],
            body=f"""Vehicle {vehicle.vehicle_id} allocated for 
            Shop {shop.name}-{shop.location} deviated at {location.lat}, {location.lng}\n
            Start Tracking: {BASE_URL}/track/{tid}
            \n\nRegards,\nSmartPDS
            """
        )
        vehicle.status = "alert_sent"

    if distance_to_destination < 0.1 and vehicle.status != "at_pds":
        send_email(
            subject=f"Vehicle {vehicle.vehicle_id} reached!",
            recipient=[shop.owner, admin.email],
            body=f"Vehicle {vehicle.vehicle_id} reached your {shop.name} - {shop.location} PDS Shop/n/nRegards,/nSmartPDS"
        )
        vehicle.status = "at_pds"

    db.session.commit()

    idle = (time.time() - location.timestamp) > 600
    eta_minutes = (distance_to_destination / 30) * 60
    estimated_arrival_time = datetime.utcfromtimestamp(time.time() + eta_minutes * 60).isoformat() + 'Z'

    socketio.emit("vehicle_update", {
        "tid": tid,
        "vehicle_id": vehicle.vehicle_id,
        "status": vehicle.status,
        "current_location": {"lat": location.lat, "lng": location.lng},
        "waypoints": shop.waypoints,
        "destination": {
            "lat": shop.destination['lat'],
            "lng": shop.destination['lng'],
            "name": shop.name,
            "location": shop.location,
        },
        "source": {
            "lat": 11.018,
            "lng": 76.925,
            "name": "Main Warehouse",
            "location": "Coimbatore",
        },
        "distance_remaining": round(distance_to_destination, 2),
        "estimated_arrival_time": estimated_arrival_time,
        "is_idle": idle,
        "deviation_alert": deviation_alert,
        "last_updated": datetime.utcfromtimestamp(location.timestamp).isoformat() + 'Z',
    })


@main.route("/add_vechile", methods=["POST"])
def add_transport():
    data = request.json

    # Required fields
    vehicle_id = data.get("vehicle_id")
    tid = data.get("tid")

    if not vehicle_id or not tid:
        return make_response({"message": "vehicle_id and track_id are required"}, 400)

    # Create Transport instance
    new_transport = Transport(
        vehicle_id=vehicle_id.upper(),
        tid=tid,
        status="at_inventory",
        waypoints=[],               # Empty waypoints list
        destination_shop_id=None    # Not assigned to any PDS shop
    )

    db.session.add(new_transport)
    db.session.commit()

    return jsonify({"message": "Vechile added", "transport_id": new_transport.id}, 201)


@main.route("/transports", methods=["GET"])
def get_transports():
    transports = Transport.query.all()

    transport_list = []
    for t in transports:
        destination_shop = None
        if t.destination_shop_id:
            shop = PDSShop.query.get(t.destination_shop_id)
            if shop:
                destination_shop = {
                    "id": shop.id,
                    "name": shop.name,
                    "location": shop.location,
                }

        transport_list.append({
            "id": t.id,
            "vehicle_id": t.vehicle_id,
            "tid": t.tid,
            "status": t.status,
            "waypoints": t.waypoints or [],
            "destination_shop": destination_shop,
        })

    return make_response({"body": transport_list}, 200)
    location = latest_locations.get('tr')
    status = "in_inventory"

    vechile = Transport.query.filter_by(tid=tid).first()

    shop = PDSShop.query.filter_by(id=vechile.destination_shop_id)

    if not location:
        status = "Unknown"
        lat = None
        lng = None
    else:
        status = vechile.status
        lat = location.get('lat')
        lng = location.get('lng')

    # Send sample or real-time data
    socketio.emit("vehicle_update", {
        "tid": tid,
        "vehicle_id": vechile.vechile_id,
        "status": status,
        "current_location": {"lat": lat, "lng": lng},
        "waypoints": shop.waypoints,
        "destination": {
            "lat": shop.destination.get('lat'),
            "lng": shop.destination.get('lat'),
            "name": shop.name,
            "location": shop.location,
        },
        "source": {
            "lat": 11.018,
            "lng": 76.925,
            "name": "Main Warehouse",
            "location": "Coimbatore",
        },
        "distance_remaining": 1.8,
        "estimated_arrival_time": "2025-05-14T11:15:00Z",
        "is_idle": False,
        "deviation_alert": False,
        "last_updated": "2025-05-14T10:42:00Z",
    })


@main.route('/stock_in_details', methods=['GET'])
def stock_in_details():
    batches = Batch.query.order_by(Batch.id.desc()).limit(5).all()
    result = []

    # Total batches
    total_batches = db.session.query(func.count(Batch.id)).scalar()

    completed_batches = db.session.query(Batch.id).join(StockItem)\
    .group_by(Batch.id)\
    .having(func.sum(StockItem.remaining_quantity) == 0).count()

    remaining_batches = total_batches - completed_batches

    for batch in batches:
        stock_items = StockItem.query.filter_by(batch_id=batch.id).all()
        items_list = []

        total_quantity = sum(item.total_quantity for item in stock_items)
        remaining_quantity = sum(item.remaining_quantity for item in stock_items)

        if total_quantity == 0:
            completion_percent = 0
        else:
            completion_percent = round((total_quantity - remaining_quantity) / total_quantity * 100, 2)

        for stock in stock_items:
            allocations = []
            related_allocations = StockAllocation.query.filter_by(stock_item_id=stock.id).order_by(StockAllocation.id.desc()).limit(5).all()

            for alloc in related_allocations:
                shop = alloc.pds_shop
                allocations.append({
                    "allocation_id": alloc.id,
                    "quantity_allocated": alloc.quantity_allocated,
                    "packs_allocated": alloc.packs_allocated,
                    "allocated_pack_codes": alloc.allocated_pack_codes,
                    "shop_id": shop.id if shop else None,
                    "shop_name": shop.name if shop else None,
                    "shop_location": shop.location if shop else None,
                })

            items_list.append({
                "stock_item_id": stock.id,
                "godown_id": stock.godown_id,
                "godown_name": stock.godown.name if stock.godown else None,
                "item_id": stock.item_id,
                "item_name": stock.item.name if stock.item else None,
                "total_quantity": total_quantity,
                "remaining_quantity": remaining_quantity,
                "num_of_packs": stock.num_of_packs,
                "current_num_of_packs": stock.current_num_of_packs,
                "pack_codes": stock.pack_codes,
                "departure_date": stock.departure_date.isoformat() if stock.departure_date else None,
                "allocations": allocations
            })

        result.append({
            "batch_id": batch.id,
            "batch_number": batch.id,
            "arrival_date": batch.arrival_date.isoformat() if batch.arrival_date else None,
            "stock_items": items_list,
            "completion_percentage": completion_percent
        })


    return jsonify({
        "body": result, 
        "total_batches": total_batches,
        "remaining_batches": remaining_batches,
        "completed_batches": completed_batches
    }), 200


@main.route('/all_godowns', methods=['GET'])
def get_all_godowns():
    godowns = Godown.query.all()
    return jsonify({'godowns': [{'id': g.id, 'name': g.name.capitalize()} for g in godowns]}), 200


@main.route('/stock_allocations', methods=['GET'])
def shop_allocations_latest():
    shops = PDSShop.query.all()
    result = []
    schedules_list = []

    for shop in shops:
        shop_data = {
            "shop_id": shop.id,
            "shop_name": shop.name,
            "location": shop.location,
            "owner": User.query.filter_by(id=shop.owner).first().email if shop.owner else None,
            "allocations": []
        }

        latest_allocations = (
            StockAllocation.query
            .filter_by(shop_id=shop.id)
            .order_by(StockAllocation.id.desc())
            .limit(5)
            .all()
        )

        for alloc in latest_allocations:
            stock_item = StockItem.query.get(alloc.stock_item_id)
            item = Item.query.get(stock_item.item_id) if stock_item else None

            shop_data["allocations"].append({
                "allocation_id": alloc.id,
                "item_name": item.name if item else "Unknown",
                "batch_id": stock_item.batch_id if stock_item else None,
                "quantity_allocated": alloc.quantity_allocated,
                "packs_allocated": alloc.packs_allocated,
                "allocated_pack_codes": alloc.allocated_pack_codes
            })

        if len(shop_data['allocations']) == 0:
            continue 

        result.append(shop_data)

    schedules = ScheduledJob.query.all()
    for schedule in schedules:
        shop = PDSShop.query.filter_by(id=schedule.shop_id).first()
        schedules_list.append({
            "shop": shop.name, 
            "frequency":schedule.frequency, 
            "run_date":schedule.run_date.strftime("%Y-%m-%d %H:%M:%S")
        })
    

    return make_response({"body": result, "schedules": schedules_list}, 200)


@main.route('/reached_pds/<string:tid>', methods=['PUT'])
def set_vehicle_to_inventory(tid):
    vehicle = Transport.query.filter_by(tid=tid).first()
    if not vehicle:
        return make_response({"message": "Vehicle not found"}, 404)

    vehicle.status = "at_inventory"
    vehicle.destination_shop_id = None
    vehicle.waypoints = None
    db.session.commit()

    return make_response({"message": f"Vehicle '{tid}' status updated to 'at_inventory'."}, 200)


@main.route('/scheduled_jobs', methods=['GET'])
def get_scheduled_jobs():
    jobs = scheduler.get_jobs()
    job_list = []

    for job in jobs:
        job_info = {
            "id": job.id,
            "next_run_time": job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job.next_run_time else None,
            "trigger": str(job.trigger),
            "args": [str(arg) for arg in job.args if not isinstance(arg, Flask)],  # Exclude app object
        }
        job_list.append(job_info)

    return make_response({"scheduled_jobs": job_list}, 200)


def start_scheduler():
    if not scheduler.running:
        scheduler.start()
