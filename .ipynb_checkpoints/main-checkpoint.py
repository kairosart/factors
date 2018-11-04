from flask import Flask, render_template
import random
app = Flask(__name__)

def factors(num):
    return [x for x in range(1, num+1) if num%x==0]

@app.route('/')
def home():
    n = random.randint(2, 10000)
    return render_template(
    # name of template
    "index.html",
    # now we pass in our variables into the template
    random_num=n, 
    )

@app.route('/factors/<int:n>', methods=['POST', 'GET'])
def factors_display(n):
	return render_template(
    # name of template
	"factors.html",
    # now we pass in our variables into the template
	number=n, 
	factors=factors(n) 
	)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

