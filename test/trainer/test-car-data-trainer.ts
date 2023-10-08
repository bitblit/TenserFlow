import { expect } from 'chai';
import * as AWS from 'aws-sdk';
import {CarDataTrainer} from "../../src/trainer/car-data-trainer";
import {Logger} from "@bitblit/ratchet/dist/common/logger";

describe('#CarDataTrainer', function () {
  it('should send a message', async () => {
    const inst: CarDataTrainer = new CarDataTrainer();
    const res: boolean = await inst.run();

    Logger.info('Result : %s', res);
    expect(res).to.be.true;
  });

});
