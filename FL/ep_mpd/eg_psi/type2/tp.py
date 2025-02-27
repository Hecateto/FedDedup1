from oblivious.ristretto import scalar
import oprf


class SemiTrustedThirdPartyType2:

    def __init__(self):
        self.secret = scalar()  # 随机标量对象

    def get_salted_data(self, client_data: oprf.oprf.data) -> oprf.oprf.data:
        return self.secret * client_data
