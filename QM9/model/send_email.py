import smtplib, ssl
from email.message import EmailMessage



def send_email(info,msg):
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "tianyao1987@gmail.com"
    receiver_email = "tianyao1987@gmail.com"
    SUBJECT = info
    TEXT = msg

    message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
    context = ssl.create_default_context()

    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, 'hygyljbupbjclxcb')
        server.sendmail(sender_email,sender_email,message)


if __name__ =='__main__':
    send_email('er','fgfg')

