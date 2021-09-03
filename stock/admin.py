from django.contrib import admin
from stock.models import Contact
# Register your models here.
admin.site.site_header = 'Stock Price Prediction Admin Panel'
admin.site.register(Contact)