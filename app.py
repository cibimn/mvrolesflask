from flask import Flask, request, jsonify, make_response, after_this_request
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import json
import random
import secrets
from datetime import datetime, timedelta
import pytz
import traceback
from openai import OpenAI
import os
from flask_cors import CORS, cross_origin
from flask_migrate import Migrate
import stripe
from dotenv import load_dotenv
from loguru import logger
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler,LoguruFormatter
import pymysql
import time

load_dotenv()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
CORS(app,expose_headers=["Authorization"])
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)
stripe.api_key = os.getenv('STRIPE_TEST_KEY')

loki_url = os.getenv("LOKI_URL")
custom_handler = LokiLoggerHandler(url=loki_url, labels={"application": "flask-app"},timeout=10, defaultFormatter=LoguruFormatter(),)
# logger.configure(handlers=[{"sink": custom_handler, "serialize": True}])
logger.add(LokiLoggerHandler(url=loki_url, labels={"application": "flask-app"},timeout=10, defaultFormatter=LoguruFormatter(),))
logger.info("Test log entry")
logger.info("Flask application has started")

@app.before_request
def before_request_func():
    request.start_time = time.time()

    @after_this_request
    def after_request_func(response):
        duration = round(time.time() - request.start_time, 2)
        logger.info(f"Method: {request.method}, Path: {request.path}, Status: {response.status_code}, Duration: {duration}s")
        return response

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    credits = db.Column(db.Integer, default=0)
    auth_token = db.Column(db.String(120),unique=True, nullable=True)
    token_created_at = db.Column(db.DateTime, nullable=True)
    country = db.Column(db.String(70), nullable=False)
    last_predict_used = db.Column(db.DateTime, nullable=True)
    is_premium = db.Column(db.Boolean, default=False)
    free_predictions_used = db.Column(db.Integer, default=0)
    
    @staticmethod
    def reset_free_predictions():
        users = User.query.all()
        for user in users:
            user.free_predictions_used = 0
        db.session.commit()
# Utility function to get IST timestamp
def get_ist_timestamp():
    utc_now = datetime.utcnow()
    ist_timezone = pytz.timezone('Asia/Kolkata')
    ist_time = utc_now.astimezone(ist_timezone)
    return ist_time

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(5000), nullable=False)
    user_email = db.Column(db.String(120), db.ForeignKey('user.email'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, prediction, user_email):
        self.prediction = prediction
        self.user_email = user_email
        self.timestamp = get_ist_timestamp()

class CreditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), db.ForeignKey('user.email'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    credits = db.Column(db.Integer, default=0)
    transaction_id = db.Column(db.String(200), nullable=False)
    
    def __init__(self, credits, transaction_id, user_email):
        self.credits = credits
        self.transaction_id = transaction_id
        self.user_email = user_email
        self.timestamp = get_ist_timestamp()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feedback = db.Column(db.String(300), nullable=False)
    prediction = db.Column(db.String(200), nullable=False)
    user_email = db.Column(db.String(120), db.ForeignKey('user.email'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, prediction, feedback, user_email):
        self.prediction = prediction
        self.feedback = feedback
        self.user_email = user_email
        self.timestamp = get_ist_timestamp()

with app.app_context():
    db.create_all()

def load_answers_data():
    with open('pred.json', 'r') as file:
        return json.load(file)

answers_data = load_answers_data()

@app.route('/get_random_answer', methods=['GET'])
def get_random_answer():
    random_key = random.choice(list(answers_data.keys()))  # Use choice to get a random key
    random_answer = answers_data.get(str(random_key), "No answer found")
    return jsonify({'prediction': random_answer})

@app.route('/profile', methods=['GET'])
def get_profile():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            logger.warning("Authorization header is missing", code=401)
            return jsonify({'message': 'Authorization header is missing'}), 401

        user = User.query.filter_by(auth_token=auth_token).first()
        if not user:
            logger.warning("Invalid or expired auth token", code=401)
            return jsonify({'message': 'Invalid or expired auth token'}), 401

        profile_data = {
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
            'country': user.country
        }

        return jsonify({'profile': profile_data}), 200
    except Exception as e:
        logger.error(f"Error in get_profile route: {str(e)}")
        return jsonify({'message': 'An error occurred while fetching the profile details'}), 500

@app.route('/update-profile', methods=['PUT'])
def update_profile():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401

        data = request.get_json()
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        country = data.get('country')

        user = User.query.filter_by(auth_token=auth_token).first()
        if not user:
            logger.warning("Invalid or expired auth token", code=401)
            return jsonify({'message': 'Invalid or expired auth token'}), 401

        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if country:
            user.country = country
        db.session.commit()

        return jsonify({'message': 'Profile updated successfully'}), 200
    except Exception as e:
        logger.error(f"Error in update_profile route: {str(e)}")
        return jsonify({'message': 'An error occurred while updating the profile'}), 500

@app.route('/change-password', methods=['POST'])
def change_password():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401

        data = request.get_json()
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        user = User.query.filter_by(auth_token=auth_token).first()
        if not user:
            logger.warning("Invalid or expired auth token", code=401)
            return jsonify({'message': 'Invalid or expired auth token'}), 401

        if not bcrypt.check_password_hash(user.password, old_password):
            logger.warning("Incorrect old password", code=401)
            return jsonify({'message': 'Incorrect old password'}), 401

        user.password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        db.session.commit()

        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        logger.error(f"Error in change_password route: {str(e)}")
        return jsonify({'message': 'An error occurred while changing the password'}), 500


@app.route('/transaction-history', methods=['GET'])
def transaction_history():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            if user.token_created_at and datetime.utcnow() > (user.token_created_at + timedelta(hours=1)):
                user.auth_token = None
                user.token_created_at = None
                db.session.commit()
                logger.warning("Token has expired. You are logged out.", code=401)
                return jsonify({'message': 'Token has expired. You are logged out.'}), 401
            else:
                transactions = CreditLog.query.filter_by(user_email=user.email).all()
                transaction_data = [
                    {'transaction_id': trans.transaction_id, 'credits': trans.credits, 'date': trans.timestamp}
                    for trans in transactions
                ]
                return jsonify({'transactions': transaction_data}), 200
        else:
            logger.warning("Invalid or expired auth token", code=401)
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in check_token route: {str(e)}")
        return jsonify({'message': 'An error occurred while checking the token'}), 500

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    data = request.get_json()
    credit_plan = str(data.get('credit_plan'))
    print(credit_plan,type(credit_plan))
    # Define your Stripe price IDs
    price_ids = {
        '10': os.getenv('PRICE_ID_FOR_10'),
        '50': os.getenv('PRICE_ID_FOR_50'),
        '100': os.getenv('PRICE_ID_FOR_100'),
    }

    price_id = price_ids.get(credit_plan)
    if not price_id:
        return jsonify({'message': 'Invalid credit plan selected'}), 400

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='payment',
            success_url=os.getenv('FRONT_END_DOMAIN')+'/payment-success/?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=os.getenv('FRONT_END_DOMAIN')+'/payment-cancel',
        )
        return jsonify({'id': checkout_session.id})
    except Exception as e:
        print(e)
        return jsonify(error=str(e)), 403

@app.route('/payment-success', methods=['POST'])
def payment_success():
    data = request.get_json()
    session_id = data.get('session_id')
    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            amount_paid = session.amount_total / 100
            print(amount_paid)
            if amount_paid == 170.0:
                credits_purchased = 10
            elif amount_paid == 808.00:
                credits_purchased = 50
            elif amount_paid == 1530.00:
                credits_purchased = 100

            auth_header = request.headers.get('Authorization')
            if auth_header:
                auth_token = auth_header.split(" ")[1]
            else:
                return jsonify({'message': 'Authorization header is missing'}), 401
            user = User.query.filter_by(auth_token=auth_token).first()
            if user:
                existing_transaction = CreditLog.query.filter_by(transaction_id=session.payment_intent).first()
                if existing_transaction:
                    return jsonify({'message': 'Transaction already processed'}), 200
                user.credits += int(credits_purchased)
                user.is_premium = True
                db.session.commit()

                new_credit_log = CreditLog(
                    user_email=user.email,
                    credits=int(credits_purchased),
                    transaction_id=session.payment_intent 
                )
                db.session.add(new_credit_log)
                db.session.commit()

                return jsonify({'message': 'Payment succeeded, credits added'}), 200
            else:
                logger.warning("User not found", code=404)
                return jsonify({'message': 'User not found'}), 404

        except Exception as e:
            return jsonify({'message': str(e)}), 500
    else:
        return jsonify({'message': 'No session ID provided'}), 400

@app.route('/cancel', methods=['GET'])
def payment_cancel():
    return jsonify({'message': 'Payment cancelled by the user'}), 200

def generate_prompt(country):
    try:
        prompt = (
            f"Imagine a fictional character from {country} in a parallel universe. "
            "Describe their profession in detail, including their name, the nature of their job, "
            "and a specific activity they are currently engaged in that is characteristic of their profession. "
            "The description should be suitable for creating a comic book-style illustration. "
            "Format the response as: 'Your variant [name] is a [profession], currently [detailed activity related to the profession].'"
        )
        response = client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150
            )
        return True,response.choices[0].text.strip()
    except:
        return False,None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            
            if user.last_predict_used and user.last_predict_used.date() == datetime.utcnow().date() and user.credits <= 0:
                if user.is_premium:
                    logger.warning("User not found", code=401)
                    return jsonify({'message': 'No credits available. Purchase More Credits'}), 403
                return jsonify({'message': 'Predict function already used today. Purchase Credits Today.'}), 403
            if user.credits > 0:
                user.credits -= 1
            country = user.country
            status,prediction = generate_prompt(country)
            if not status:
                random_key = random.choice(list(answers_data.keys()))
                prediction = answers_data.get(str(random_key), "No answer found")
            new_log = Log(prediction=str(prediction), user_email=user.email)
            user.last_predict_used = datetime.utcnow()
            db.session.add(new_log)
            db.session.commit()
            if prediction != "No answer found":
                response = client.images.generate(model="dall-e-3",prompt=prediction,size="1024x1024",quality="standard",n=1,)
                print(response)
                imageurl = response.data[0].url
            else:
                imageurl = None
            
            return jsonify({'prediction': prediction,"pred_image_url":imageurl}), 200
        else:
            logger.warning("Invalid auth token", code=401)
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in log route: {str(e)}")
        return jsonify({'message': 'An error occurred while storing the prediction'}), 500

@app.route('/check_token', methods=['POST'])
def check_token():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            if user.token_created_at and datetime.utcnow() > (user.token_created_at + timedelta(hours=1)):
                user.auth_token = None
                user.token_created_at = None
                db.session.commit()
                logger.warning("Token has expired. You are logged out.", code=401)
                return jsonify({'message': 'Token has expired. You are logged out.'}), 401
            else:
                return jsonify({'message': 'Token is valid.'}), 200
        else:
            logger.warning("Invalid auth token", code=401)
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in check_token route: {str(e)}")
        return jsonify({'message': 'An error occurred while checking the token'}), 500

@app.route('/get_credits', methods=['POST'])
def get_credits():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            if user.token_created_at and datetime.utcnow() > (user.token_created_at + timedelta(hours=1)):
                user.auth_token = None
                user.token_created_at = None
                
                db.session.commit()
                logger.warning("Token has expired. You are logged out.", code=401)
                return jsonify({'message': 'Token has expired. You are logged out.'}), 401
            else:
                credits = user.credits
                return jsonify({'credits': credits}), 200
        else:
            logger.warning("Invalid auth token", code=401)
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in check_token route: {str(e)}")
        return jsonify({'message': 'An error occurred while checking the token'}), 500

@app.route('/login', methods=['POST'])
@cross_origin(expose_headers=["Authorization"])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            auth_token = secrets.token_hex(16)
            user.auth_token = auth_token
            user.token_created_at = datetime.utcnow()
            db.session.commit()
            response = jsonify({'message': 'Login successful'})
            response.headers['Authorization'] = f'Bearer {auth_token}'
            return response
        else:
            return jsonify({'message': 'Invalid login credentials'}), 401
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in login route: {str(e)}")
        return jsonify({'message': 'An error occurred during login'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    try:
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            return jsonify({'message': 'Authorization header is missing'}), 401
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            user.auth_token = None
            db.session.commit()
            return jsonify({'message': 'Logout successful'})
        else:
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in logout route: {str(e)}")
        return jsonify({'message': 'An error occurred during logout'}), 500


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    required_fields = ['first_name', 'last_name', 'email', 'password', 'country']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'message': f'Missing fields: {", ".join(missing_fields)}'}), 400
    
    first_name = data['first_name']
    last_name = data['last_name']
    email = data['email']
    password = data['password']
    country = data['country']
    credit = 0
    
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'message': 'Email already registered'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    new_user = User(first_name=first_name, last_name=last_name, email=email, password=hashed_password, credits=credit, country=country)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'Registration successful'}), 200

@app.route('/log', methods=['POST'])
def log():
    try:
        data = request.get_json()
        auth_token = data.get('auth_token')
        prediction = data.get('prediction')
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            new_log = Log(prediction=prediction, user_email=user.email)
            db.session.add(new_log)
            db.session.commit()
            return jsonify({'message': 'Log saved successfully'}), 200
        else:
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in store_prediction route: {str(e)}")
        return jsonify({'message': 'An error occurred while storing the prediction'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        auth_token = data.get('auth_token')
        feedback = data.get('feedback')
        prediction = data.get('prediction')
        user = User.query.filter_by(auth_token=auth_token).first()
        if user:
            new_feedback = Feedback(prediction=prediction, feedback=feedback, user_email=user.email)
            db.session.add(new_feedback)
            db.session.commit()
            return jsonify({'message': 'Prediction saved successfully'}), 200
        else:
            return jsonify({'message': 'Invalid auth token'}), 401
    except Exception as e:
        logger.error(f"Error in store_prediction route: {str(e)}")
        return jsonify({'message': 'An error occurred while storing the prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
