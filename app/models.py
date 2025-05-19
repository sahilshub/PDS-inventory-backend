from app.extensions import db
from datetime import datetime


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(520), nullable=False)
    verified = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(100), default='public')


class Blockchain(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    data = db.Column(db.Text, nullable=False)
    previous_hash = db.Column(db.String(100), nullable=False)
    hash = db.Column(db.String(100), nullable=False)  


class Batch(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    arrival_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class Godown(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f"<Godown(id={self.id}, name={self.name})>"


class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

    def __repr__(self):
        return f"<Item(id={self.id}, name={self.name})>"


class PDSShop(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    owner = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    destination = db.Column(db.JSON)  # Dict: {"lat": ..., "lng": ...}
    waypoints = db.Column(db.JSON)  # List[Dict[str, floa

    user = db.relationship('User', backref='pds_owners')

    def __repr__(self):
        return f"<PDSShop(id={self.id}, name={self.name})>"
    

class ShopRequirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.Integer, db.ForeignKey('pds_shop.id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    required_quantity = db.Column(db.Integer, nullable=False)

    pds_shop = db.relationship('PDSShop', backref='requirements')
    item = db.relationship('Item')


class StockItem(db.Model):
    __tablename__ = 'stock_items'

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'))
    godown_id = db.Column(db.Integer, db.ForeignKey('godown.id'))
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'))

    total_quantity = db.Column(db.Float)
    remaining_quantity = db.Column(db.Float)

    num_of_packs = db.Column(db.Integer)  # Total packs at stock-in
    current_num_of_packs = db.Column(db.Integer)  # Current remaining packs

    pack_codes = db.Column(db.String(30), nullable=True) 
    
    departure_date = db.Column(db.DateTime, nullable=True)

    batch = db.relationship('Batch', backref='items')
    godown = db.relationship('Godown', backref='godown_items')
    item = db.relationship('Item')


class StockAllocation(db.Model):
    __tablename__ = 'stock_allocations'
    
    id = db.Column(db.Integer, primary_key=True)
    stock_item_id = db.Column(db.Integer, db.ForeignKey('stock_items.id'))
    shop_id = db.Column(db.Integer, db.ForeignKey('pds_shop.id'))

    quantity_allocated = db.Column(db.Float)
    packs_allocated = db.Column(db.Integer)

    allocated_pack_codes = db.Column(db.PickleType, default=[])
    
    pds_shop = db.relationship('PDSShop', backref='allocations')


class ScheduledJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.Integer, db.ForeignKey('pds_shop.id'), nullable=False)
    run_date = db.Column(db.DateTime, nullable=False)  # First run time
    frequency = db.Column(db.String(20), nullable=False)  # 'daily', 'weekly', 'monthly'

    shop = db.relationship('PDSShop', backref='scheduled_jobs')
    

class Transport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.String(100), unique=True, nullable=False)
    tid = db.Column(db.String(100), unique=True, nullable=False)  # For OwnTracks device
    waypoints = db.Column(db.JSON)  # List[Dict[str, float]]: [{"lat": ..., "lng": ...}]
    destination_shop_id = db.Column(db.Integer, db.ForeignKey('pds_shop.id'))
    status = db.Column(db.String(20), default="at_inventory")  # "on_the_way", "at_pds"
    last_seen = db.Column(db.DateTime)

    pds_shop = db.relationship("PDSShop", backref="transport")  


class VehicleLocation(db.Model):
    __tablename__ = 'vehicle_locations'

    tid = db.Column(db.String, primary_key=True)
    lat = db.Column(db.Float, nullable=False)
    lng = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.Float, nullable=False)


