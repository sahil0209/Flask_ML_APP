from flask_wtf import FlaskForm
from wtforms.fields import FloatField, SubmitField
from wtforms.validators import DataRequired

class IrisForm(FlaskForm):
    sepal_length = FloatField('Sepal Length',validators=[DataRequired()])
    sepal_width = FloatField('Sepal Width',validators=[DataRequired()])
    petal_length = FloatField('Petal Length',validators=[DataRequired()])
    petal_width = FloatField('Petal Width',validators=[DataRequired()])
    submit = SubmitField('Predict')