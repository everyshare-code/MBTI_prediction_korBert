from flask import Flask
from flask_restful import Api
from api.recommend_api import RecommendBySearch
from api.predict_api import PredictAPI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


# 엔드포인트 추가
api.add_resource(RecommendBySearch, '/recommend_by_search','/recommend_by_mbti')
api.add_resource(PredictAPI, '/predict_mbti')

if __name__ == '__main__':
    app.run(debug=True,port=5000)
