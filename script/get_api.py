import get_api_entry as ae
import get_api_hr as ah
import get_api_jk as aj
import get_api_tr as at
import get_api_train as atrain


def get_data(meet, date):
    ae.get_data(meet, date)
    ah.get_data(meet)
    aj.get_data(meet)
    at.get_data(meet)
    atrain.get_data(meet, date)


if __name__ == '__main__':
    get_data(2, 20161111)


